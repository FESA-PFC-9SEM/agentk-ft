"""The "mcp" tool layer: demo/mcp_server.py's ToolService behind
demo/mcp_client.py's McpToolbox. Most tests connect the two through a fake
connection that serializes every call as JSON (no subprocess, no cluster);
one launches the real server over stdio."""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import yaml

from demo.agent import TOOLS, Agent, Toolbox
from demo.mcp_server import HOST_TOOLS, MODEL_TOOLS, ToolService
from demo.test_agent import AUDIT, call, scripted
from demo.test_agent_tools import DEPLOYMENT, NEW_CONFIGMAP, FakeKube, toolbox_for

pytest.importorskip("mcp")

from demo.audit_log import Actor  # noqa: E402
from demo.mcp_client import McpToolbox  # noqa: E402


class JsonConnection:
    """Stands in for McpConnection: tools/list and tools/call, with every
    argument and result passed through JSON like the real transport."""

    def __init__(self, service: ToolService):
        self.service, self.calls, self.closed = service, [], False

    def list_tools(self):
        return [SimpleNamespace(name=t["name"], description=t["description"], input_schema=t["parameters"]) for t in MODEL_TOOLS + HOST_TOOLS]

    def call_tool(self, name, arguments):
        self.calls.append(name)
        return self.service.handle(name, json.loads(json.dumps(arguments)))

    def close(self):
        self.closed = True


class FakeAuditLog:
    def __init__(self):
        self.records = []

    def ping(self):
        return True

    def record(self, actor, adapter, record):
        self.records.append((actor, adapter, record))
        return len(self.records)


def mcp_layer(kube, audit_result=AUDIT, log=None):
    server_toolbox = toolbox_for(kube)
    server_toolbox.can_apply = False  # as build_toolbox(): a viewer until the host says otherwise
    service = ToolService(server_toolbox, audit_log_factory=lambda: log)
    host = McpToolbox(audit=lambda text: (json.loads(json.dumps(audit_result)), []), connection=JsonConnection(service))
    return host, server_toolbox


def operator(host, log=None):
    host.set_session(True, Actor("sub-1", "operator", None, ("viewer", "operator")), "runs/x/lora_adapter", log)


def test_the_planner_sees_the_same_tools_in_both_layers():
    host, _ = mcp_layer(FakeKube())
    assert [t["function"]["name"] for t in host.tools] == [t["function"]["name"] for t in TOOLS]
    assert host.tools == TOOLS  # same definitions, discovered from the server
    assert not any(t["function"]["name"].startswith("host_") for t in host.tools)


def test_audit_runs_in_the_host_and_apply_fix_in_the_server():
    kube = FakeKube([DEPLOYMENT])
    host, server_toolbox = mcp_layer(kube)
    operator(host)
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "I prepared a fix.",
    )
    agent = Agent(planner, host)
    events = list(agent.ask("check web in demo and fix it"))
    assert [e.type for e in events] == ["tool_call", "tool_result", "tool_call", "confirm", "final"]
    assert "1 problem(s)" in events[1].text
    assert agent.waiting and server_toolbox.pending is not None and kube.applied() == []
    assert host.pending is host.pending  # one mirror object per change (the UI's widget keys depend on it)
    list(agent.resume(approved=True))
    assert "nginx:1.25" in kube.applied()[-1][1] and server_toolbox.pending is None


def test_the_server_session_is_a_viewer_until_the_host_says_otherwise():
    kube = FakeKube([DEPLOYMENT])
    host, _ = mcp_layer(kube)
    out = host.call("scale_resource", {"kind": "Deployment", "name": "web", "namespace": "demo", "replicas": 3})
    assert out.startswith("Refused") and "viewer" in out
    operator(host)
    assert "from 2 to 3" in host.call("scale_resource", {"kind": "Deployment", "name": "web", "namespace": "demo", "replicas": 3})


def test_preview_confirm_and_the_audit_log_happen_in_the_server():
    kube, log = FakeKube([DEPLOYMENT]), FakeAuditLog()
    host, _ = mcp_layer(kube, log=log)
    operator(host, log)
    host.last_request = "scale web to 3"
    host.call("scale_resource", {"kind": "Deployment", "name": "web", "namespace": "demo", "replicas": 3})
    change = host.pending
    preview, error = host.preview(change)
    assert error is None and yaml.safe_load(preview)["spec"]["replicas"] == 3
    assert host.audit_log.ping()
    result = host.execute_pending()
    assert result.startswith("Applied") and "(audit log: change #1)" in result
    actor, adapter, record = log.records[0]
    assert (actor.username, adapter, record["action"], record["user_request"]) == ("operator", "runs/x/lora_adapter", "scale", "scale web to 3")


def test_a_stale_confirm_is_refused():
    kube = FakeKube([DEPLOYMENT])
    host, _ = mcp_layer(kube)
    operator(host)
    host.call("apply_yaml_to_cluster", {"manifest": NEW_CONFIGMAP, "namespace": "demo"})
    stale = host.pending
    host.decline_pending()
    assert host.connection.service.handle("host_confirm_change", {"change_id": stale.change_id})[0] == "That change is no longer pending."
    assert kube.applied() == []


def test_without_an_audit_log_the_status_says_so():
    host, _ = mcp_layer(FakeKube())
    host.set_session(True, None, None, None)
    assert host.audit_log is None


def test_unknown_and_bad_calls():
    host, _ = mcp_layer(FakeKube())
    assert host.call("format_disk", {}).startswith("Unknown tool")
    assert host.call("audit_manifest", {"pod": "x"}).startswith("Bad arguments")
    text, is_error = host.connection.service.handle("host_preview_change", {})
    assert is_error and text.startswith("Bad arguments")


def test_build_agent_rejects_an_unknown_layer():
    from demo.agent import build_agent

    with pytest.raises(ValueError):
        build_agent(None, None, tool_layer="grpc")


# ------------------------------------------- the real server, over stdio --


def test_the_real_server_over_stdio():
    """Launches demo/mcp_server.py and talks MCP to it -- protocol, discovery
    and a guardrail that refuses before any kubectl call (no cluster needed)."""
    from demo.mcp_client import REPO_ROOT, McpConnection

    connection = McpConnection(sys.executable, ["-m", "demo.mcp_server"], env=dict(os.environ), cwd=str(REPO_ROOT), timeout=60)
    try:
        assert connection.server_info.name == "k8s-cluster-agent-tools"
        names = {t.name for t in connection.list_tools()}
        assert {t["function"]["name"] for t in TOOLS} - names == {"audit_manifest"}
        text, is_error = connection.call_tool("list_k8s_resource_names", {"resource_type": "pods", "namespace": "audit"})
        assert text.startswith("Refused") and not is_error
        text, _ = connection.call_tool("host_pending_change", {})
        assert json.loads(text) is None
    finally:
        connection.close()


def test_server_toolbox_defaults():
    from demo.mcp_server import build_toolbox

    toolbox = build_toolbox()
    assert isinstance(toolbox, Toolbox) and toolbox.can_apply is False and "audit" in toolbox.protected_namespaces
