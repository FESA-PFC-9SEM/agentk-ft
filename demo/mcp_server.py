"""
MCP server for the cluster agent's tools -- the "mcp" tool layer
(demo/run_secure.sh --tools mcp). The default "function" layer runs the
same tools in-process (demo/agent.py's Toolbox); this server runs them in a
separate process and serves them over the Model Context Protocol (JSON-RPC,
stdio transport), using the official MCP Python SDK.

What it serves:
  - the model's tools, with the same names, descriptions and JSON Schemas
    as the in-process layer: list_k8s_resource_names,
    extract_k8s_resource_yamls, get_specific_object_yaml,
    validate_k8s_yaml_dry_run, apply_yaml_to_cluster, get_pod_logs,
    get_pod_diagnostics, scale_resource, apply_fix and run_kubectl. They run
    kubectl here, as the agent's service account (AGENT_KUBECONFIG), with
    every guardrail of the in-process layer -- the code is the same Toolbox;
  - host tools (prefixed "host_"), for the app that launched it and never
    shown to the model: the audit hand-off, the session's identity and role,
    and the pending change's preview/confirm/decline.

audit_manifest isn't a server tool: it needs the fine-tuned model, which
lives in the host -- the MCP host owns the model; the sampling capability
that let a server borrow the client's model is deprecated (SEP-2577). The
host fetches the manifest here (host_audit_input), runs the model, and hands
the result back (host_record_audit), so apply_fix finds it here.

The pending change and the audit log live here too: a change is proposed by
a model tool, and only host_confirm_change applies it -- after the user
clicks Confirm in the app -- recording it like the in-process layer does.

Trust: over stdio, the server's only client is the process that launched it,
and that client asserts the user's identity and role (host_set_session),
taken from its Keycloak login. Until it does, the session is a viewer: no
change can be confirmed. A network transport would need MCP's OAuth instead.

Usage (the app starts it itself in the mcp layer):
    finetune/.venv/bin/python -m demo.mcp_server
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.agent import TOOLS, Toolbox  # noqa: E402

# The model's tools: everything but audit_manifest, which runs in the host.
MODEL_TOOLS = [t["function"] for t in TOOLS if t["function"]["name"] != "audit_manifest"]
HOST_PREFIX = "host_"


def _host_tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"name": HOST_PREFIX + name, "description": "[host only, not for the model] " + description,
            "parameters": {"type": "object", "properties": properties, "required": required or []}}


_TARGET = {"kind": {"type": "string"}, "name": {"type": "string"}, "namespace": {"type": "string"}}
HOST_TOOLS = [
    _host_tool("set_session", "Set the user's identity and role, the model adapter, whether changes are recorded in the "
               "audit log, and the user's latest request. Only the given fields change.",
               {"can_apply": {"type": "boolean"}, "actor": {"type": "object"}, "adapter": {"type": "string"},
                "audit": {"type": "boolean"}, "last_request": {"type": "string"}}),
    _host_tool("audit_input", "The manifest to audit, as JSON {manifest} or {error}.", _TARGET, ["kind", "name"]),
    _host_tool("record_audit", "Keep the host's audit result (apply_fix uses it); returns its description for the model.",
               {**_TARGET, "manifest": {"type": "string"}, "parsed": {"type": ["object", "null"]},
                "errors": {"type": "array", "items": {"type": "string"}}}, ["kind", "name", "manifest"]),
    _host_tool("ready_fixes", "Audits with a fix ready, as a JSON list of [kind, name, namespace].", {}),
    _host_tool("pending_change", "The change waiting for the user's confirmation, as JSON (null if none).", {}),
    _host_tool("preview_change", "The pending change's manifest with the user's image choices, as JSON {yaml, error}.",
               {"change_id": {"type": "string"}, "image_overrides": {"type": "object"}}, ["change_id"]),
    _host_tool("confirm_change", "The user clicked Confirm: apply the pending change and record it.",
               {"change_id": {"type": "string"}, "new_resources": {"type": ["array", "null"], "items": {"type": "string"}},
                "image_overrides": {"type": "object"}}, ["change_id"]),
    _host_tool("decline_change", "The user clicked Decline: drop the pending change and record it.",
               {"change_id": {"type": "string"}}, ["change_id"]),
    _host_tool("audit_log_status", "Whether changes are recorded, and whether the audit log is reachable, as JSON.", {}),
]


def build_toolbox() -> Toolbox:
    from demo import core

    return Toolbox(
        audit=lambda text: (None, ["audits run in the host (MCP host), not in the server"]),
        kubectl=core.kubectl,
        list_resources=core.list_resources,
        fetch_yaml=core.fetch_resource_yaml,
        apply_patch=core.apply_patch_to_manifest,
        protected_namespaces=frozenset(core.PROTECTED_NAMESPACES),
        created_label=core.AGENT_LABEL,
        can_apply=False,  # a viewer until the host says otherwise
    )


class ToolService:
    """The tools' logic, transport-free: name + arguments -> (text, is_error)."""

    def __init__(self, toolbox: Toolbox, audit_log_factory=None):
        self.toolbox = toolbox
        self._audit_log_factory = audit_log_factory or _default_audit_log
        self._model_tools = {t["name"] for t in MODEL_TOOLS}
        self._lock = threading.Lock()

    def handle(self, name: str, arguments: dict) -> tuple[str, bool]:
        with self._lock:
            if name in self._model_tools:
                return self.toolbox.call(name, arguments), False
            handler = getattr(self, name.removeprefix(HOST_PREFIX), None) if name.startswith(HOST_PREFIX) else None
            if handler is None:
                return f"Unknown tool '{name}'.", True
            try:
                return handler(**arguments), False
            except (TypeError, ValueError, KeyError) as e:
                return f"Bad arguments for {name}: {e}", True

    # ---- host tools -------------------------------------------------------

    def set_session(self, can_apply=None, actor=None, adapter=None, audit=None, last_request=None) -> str:
        from demo.audit_log import Actor

        tb = self.toolbox
        if can_apply is not None:
            tb.can_apply = bool(can_apply)
        if actor is not None:
            tb.actor = Actor(str(actor["subject"]), str(actor["username"]), actor.get("email"), tuple(actor.get("roles") or ()))
        if adapter is not None:
            tb.adapter = adapter
        if audit is not None:
            tb.audit_log = self._audit_log_factory() if audit else None
        if last_request is not None:
            tb.last_request = last_request
        return "ok"

    def audit_input(self, kind, name, namespace=None) -> str:
        manifest, problem = self.toolbox.audit_input(kind, name, namespace)
        return json.dumps({"error": problem} if problem else {"manifest": manifest})

    def record_audit(self, kind, name, manifest, namespace=None, parsed=None, errors=None) -> str:
        return self.toolbox.record_audit(kind, name, namespace, manifest, parsed, list(errors or []))

    def ready_fixes(self) -> str:
        return json.dumps(self.toolbox.ready_fixes())

    def pending_change(self) -> str:
        change = self.toolbox.pending
        return json.dumps(change.to_dict() if change else None)

    def _check(self, change_id: str) -> str | None:
        pending = self.toolbox.pending
        if pending is None or pending.change_id != change_id:
            return "That change is no longer pending."
        return None

    def preview_change(self, change_id, image_overrides=None) -> str:
        if (stale := self._check(change_id)):
            return json.dumps({"yaml": None, "error": stale})
        patched, error = self.toolbox.preview(self.toolbox.pending, image_overrides or {})
        return json.dumps({"yaml": patched, "error": error})

    def confirm_change(self, change_id, new_resources=None, image_overrides=None) -> str:
        return self._check(change_id) or self.toolbox.execute_pending(new_resources, image_overrides or {})

    def decline_change(self, change_id) -> str:
        return self._check(change_id) or self.toolbox.decline_pending()

    def audit_log_status(self) -> str:
        log = self.toolbox.audit_log
        return json.dumps({"enabled": log is not None, "reachable": bool(log and log.ping())})


def _default_audit_log():
    from demo.audit_log import AuditLog, database_url

    url = database_url()
    return AuditLog(url) if url else None


def create_server(service: ToolService) -> Server:
    """The MCP server: tools/list and tools/call, both answered by `service`."""

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=t["name"], description=t["description"], input_schema=t["parameters"]) for t in MODEL_TOOLS + HOST_TOOLS
        ])

    async def call_tool(ctx, params) -> types.CallToolResult:
        # The tools are blocking (kubectl, PostgreSQL): run them off the event loop.
        text, is_error = await anyio.to_thread.run_sync(service.handle, params.name, dict(params.arguments or {}))
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=is_error)

    return Server("k8s-cluster-agent-tools", version="1.0", on_list_tools=list_tools, on_call_tool=call_tool,
                  instructions="Kubernetes cluster tools of the TCC cluster agent. Changes wait for the user's confirmation in the host.")


async def _serve() -> None:
    server = create_server(ToolService(build_toolbox()))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    anyio.run(_serve)


if __name__ == "__main__":
    main()
