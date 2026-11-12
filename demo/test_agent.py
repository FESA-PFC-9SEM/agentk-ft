import json

import yaml

from demo.agent import TOOLS, Agent, Toolbox, check_read_only, parse_tool_calls
from demo.core import apply_patch_to_manifest

MANIFEST = yaml.safe_dump(
    {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "web", "namespace": "demo"},
        "spec": {"template": {"spec": {"containers": [{"name": "app", "image": "nginx"}]}}},
    },
    sort_keys=False,
)
AUDIT = {
    "findings": [
        {"rule_id": "KSEC-005", "severity": "medium", "doc": 0, "path": "/spec/template/spec/containers/0/image",
         "message": "Container image has no pinned tag: nginx", "evidence": "ngin***"}
    ],
    "patch": [{"doc": 0, "op": "replace", "path": "/spec/template/spec/containers/0/image", "value": "nginx:1.25"}],
    "new_resources": [],
    "notes": [],
}


def call(tool, **arguments):
    return f'<tool_call>\n{json.dumps({"name": tool, "arguments": arguments})}\n</tool_call>'


class FakeCluster:
    def __init__(self, dry_run_ok=True):
        self.commands = []
        self.dry_run_ok = dry_run_ok

    def kubectl(self, *args, input_text=None, timeout=None):
        self.commands.append((args, input_text))
        if "--dry-run=server" in args:
            return (True, "deployment.apps/web configured (server dry run)") if self.dry_run_ok else (False, "admission webhook denied")
        if args[:1] == ("apply",):
            return True, "deployment.apps/web configured"
        return True, "NAME   READY\nweb    1/1"

    def applied(self):
        return [c for c in self.commands if c[0][:1] == ("apply",) and "--dry-run=server" not in c[0]]


def make_toolbox(cluster, audit_result=AUDIT):
    return Toolbox(
        audit=lambda text: (json.loads(json.dumps(audit_result)), []),
        kubectl=cluster.kubectl,
        list_resources=lambda: ([{"kind": "Deployment", "namespace": "demo", "name": "web"}], None),
        fetch_yaml=lambda kind, namespace, name: (MANIFEST, None) if name == "web" else (None, f"{kind} {name} not found"),
        apply_patch=apply_patch_to_manifest,
    )


def scripted(*answers):
    """A fake planner that returns the given answers in order and records
    every conversation it was shown."""
    seen = []

    def planner(messages, tools, prefill=""):
        seen.append([dict(m) for m in messages])
        planner.prefills.append(prefill)
        assert tools is TOOLS
        return answers[min(len(seen), len(answers)) - 1]  # repeats the last answer when exhausted

    planner.seen = seen
    planner.prefills = []
    return planner


def test_parse_tool_calls_native_qwen_format():
    calls, text, errors = parse_tool_calls("Let me check.\n" + call("list_resources", namespace="demo"))
    assert calls == [{"name": "list_resources", "arguments": {"namespace": "demo"}}]
    assert text == "Let me check." and errors == []
    calls, _, _ = parse_tool_calls('<tool_call>{"name": "get_manifest", "arguments": "{\\"kind\\": \\"Pod\\", \\"name\\": \\"x\\"}"}</tool_call>')
    assert calls[0]["arguments"] == {"kind": "Pod", "name": "x"}
    calls, _, errors = parse_tool_calls("<tool_call>{not json}</tool_call>")
    assert calls == [] and len(errors) == 1


def test_read_only_guardrail():
    assert check_read_only("kubectl get pods -n demo")[0] == ["get", "pods", "-n", "demo"]
    assert check_read_only("logs web-123")[0] == ["logs", "web-123"]
    for command in ("kubectl delete pod x", "kubectl apply -f x.yaml", "kubectl exec -it x -- sh", "kubectl edit deploy web"):
        assert check_read_only(command)[0] is None, command
    for command in ("kubectl get secret db -o yaml", "kubectl get secrets -o json", "kubectl describe secret db", "kubectl get secret/db -o yaml"):
        assert check_read_only(command)[0] is None, command
    assert check_read_only("kubectl get secrets")[0] == ["get", "secrets"]  # names only is fine


def test_plain_answer_without_tools():
    agent = Agent(scripted("Hi! I can audit and fix your workloads."), make_toolbox(FakeCluster()))
    events = list(agent.ask("hello"))
    assert [e.type for e in events] == ["final"]
    assert events[0].text.startswith("Hi!")


def test_audit_then_fix_waits_for_confirmation_then_applies_and_verifies():
    cluster = FakeCluster()
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "I prepared a fix: the image gets pinned to nginx:1.25. Confirm to apply it.",
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        "Fixed: the image is now pinned to nginx:1.25.",
    )
    agent = Agent(planner, make_toolbox(cluster))
    events = list(agent.ask("check deployment web in demo and fix it"))
    assert [e.type for e in events] == ["tool_call", "tool_result", "tool_call", "confirm", "final"]
    assert "1 problem(s)" in events[1].text and "KSEC-005" in events[1].text
    # The planner saw the proposed change, so it could explain it.
    assert "nginx -> nginx:1.25" in events[3].text and "KSEC-005" in events[3].text
    assert events[-1].text.startswith("I prepared a fix")
    # Paused: dry run done, nothing applied yet.
    assert agent.waiting and cluster.applied() == []
    assert "--dry-run=server" in cluster.commands[-1][0]

    events = list(agent.resume(approved=True))
    assert [e.type for e in events] == ["tool_result", "tool_call", "tool_result", "final"]
    assert events[0].text.startswith("Applied")
    assert len(cluster.applied()) == 1 and "nginx:1.25" in cluster.applied()[0][1]
    # The planner was told the outcome before continuing.
    assert any(m["role"] == "user" and "I confirmed" in m["content"] and "Applied" in m["content"] for m in planner.seen[-1])


def test_declined_change_is_not_applied():
    cluster = FakeCluster()
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "OK, I left it unchanged.",
    )
    agent = Agent(planner, make_toolbox(cluster))
    list(agent.ask("fix web"))
    events = list(agent.resume(approved=False))
    assert events[0].text.startswith("The user declined")
    assert "Do not propose this fix again" in events[0].text
    assert cluster.applied() == [] and not agent.waiting


def test_failed_dry_run_never_asks_and_never_applies():
    cluster = FakeCluster(dry_run_ok=False)
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "The fix was rejected by the cluster.",
    )
    agent = Agent(planner, make_toolbox(cluster))
    events = list(agent.ask("fix web"))
    assert "confirm" not in [e.type for e in events]
    assert "failed server-side validation" in events[3].text
    assert cluster.applied() == []


def test_apply_fix_requires_an_audit_first():
    agent = Agent(scripted(call("apply_fix", kind="Deployment", name="web", namespace="demo"), "Let me audit it first."), make_toolbox(FakeCluster()))
    events = list(agent.ask("fix web"))
    assert "call audit_manifest first" in events[1].text


def test_new_resources_are_created_before_the_fix():
    cluster = FakeCluster()
    with_secret = dict(AUDIT, new_resources=["apiVersion: v1\nkind: Secret\nmetadata:\n  name: web-secrets\nstringData:\n  pw: real"])
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "Done.",
    )
    agent = Agent(planner, make_toolbox(cluster, with_secret))
    list(agent.ask("fix web"))
    list(agent.resume(approved=True))
    applied = cluster.applied()
    assert "kind: Secret" in applied[0][1] and "Deployment" in applied[1][1]


def test_refused_command_and_unknown_tool_are_reported_to_the_planner():
    planner = scripted(call("run_kubectl", command="kubectl delete pod web"), call("format_disk"), "I can't do that.")
    cluster = FakeCluster()
    events = list(Agent(planner, make_toolbox(cluster)).ask("delete the web pod"))
    assert events[1].text.startswith("Refused")
    assert events[3].text.startswith("Unknown tool")
    assert cluster.commands == []


def test_step_limit_stops_a_looping_planner():
    looping = scripted(*[call("list_resources")] * 10)
    events = list(Agent(looping, make_toolbox(FakeCluster()), max_steps=3).ask("loop"))
    assert events[-1].type == "final" and "stopped after 3 steps" in events[-1].text


def test_questions_while_a_fix_is_pending_keep_it_pending():
    cluster = FakeCluster()
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "I prepared a fix; confirm to apply it.",
        "It pins the image to nginx:1.25, so every restart pulls the same version.",  # answers the question
        "Applied and verified.",
    )
    agent = Agent(planner, make_toolbox(cluster))
    list(agent.ask("fix web in demo"))
    change = agent.toolbox.pending
    events = list(agent.ask("why does the fix change the image?"))
    assert [e.type for e in events] == ["final"] and "pins the image" in events[0].text
    # Still pending, same change, nothing applied -- the user can still decide.
    assert agent.waiting and agent.toolbox.pending is change and cluster.applied() == []
    # The planner answered from the change description in its conversation.
    assert any("nginx -> nginx:1.25" in str(m.get("content")) for m in planner.seen[-1])
    list(agent.resume(approved=True))
    assert len(cluster.applied()) == 1


def test_parse_tool_calls_accepts_a_json_block_naming_a_known_tool():
    # Seen from Qwen2.5-Coder on the first end-to-end run.
    text = 'Sure, let me check.\n\n```json\n{"name": "audit_manifest", "arguments": {"kind": "Deployment", "name": "web"}}\n```'
    calls, remaining, errors = parse_tool_calls(text)
    assert calls == [{"name": "audit_manifest", "arguments": {"kind": "Deployment", "name": "web"}}]
    assert remaining == "Sure, let me check." and errors == []
    bare = '{"name": "list_resources", "arguments": {}}'
    assert parse_tool_calls(bare)[0] == [{"name": "list_resources", "arguments": {}}]
    # A JSON example that isn't a call to one of our tools stays plain text.
    example = 'Example:\n```json\n{"name": "nginx", "replicas": 2}\n```'
    assert parse_tool_calls(example)[0] == []


def test_an_announced_action_is_retried_with_a_prefilled_tool_call():
    planner = scripted(
        "Sure! Let me check the web deployment for problems.",  # announces, no call
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),  # the pre-filled retry
        "It has one problem: the image isn't pinned.",
    )
    events = list(Agent(planner, make_toolbox(FakeCluster())).ask("check web in demo"))
    assert [e.type for e in events] == ["tool_call", "tool_result", "final"]
    assert planner.prefills[:2] == ["", "<tool_call>\n"]


def test_a_requested_fix_is_called_by_the_loop_after_the_audit():
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        "It has one problem: the image isn't pinned. I will now apply the fix.",  # stops short
        "I prepared the fix.",
    )
    events = list(Agent(planner, make_toolbox(FakeCluster())).ask("check web in demo and fix it"))
    assert [e.type for e in events][-2:] == ["confirm", "final"]
    assert events[-3].name == "apply_fix" and events[-3].arguments == {"kind": "deployment", "name": "web", "namespace": "demo"}


def test_a_plain_answer_is_not_nudged():
    planner = scripted("Your cluster has 3 workloads in demo.")
    events = list(Agent(planner, make_toolbox(FakeCluster())).ask("how many workloads?"))
    assert [e.type for e in events] == ["final"] and len(planner.seen) == 1


def test_a_declined_fix_is_not_proposed_again_in_the_same_turn():
    cluster = FakeCluster()
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "I prepared a fix.",
        "OK, left unchanged.",
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),  # what a nudge would cause
    )
    agent = Agent(planner, make_toolbox(cluster))
    list(agent.ask("check web in demo and fix it"))
    events = list(agent.resume(approved=False))
    assert events[-1].type == "final" and events[-1].text == "OK, left unchanged."
    assert not agent.waiting


def test_a_second_fix_is_refused_while_one_is_pending():
    cluster = FakeCluster()
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "I prepared a fix.",
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),  # tries again before the user decides
        "Please confirm or decline the pending change first.",
    )
    agent = Agent(planner, make_toolbox(cluster))
    list(agent.ask("fix web in demo"))
    change = agent.toolbox.pending
    events = list(agent.ask("also fix it again"))
    assert "already waiting for the user's confirmation" in events[1].text
    assert agent.toolbox.pending is change and "confirm" not in [e.type for e in events]


def test_unclosed_tool_call_at_the_end_is_accepted():
    calls, _, errors = parse_tool_calls('<tool_call>\n{"name": "list_resources", "arguments": {}}')
    assert calls == [{"name": "list_resources", "arguments": {}}] and errors == []


def test_a_suggested_kubectl_command_is_run_instead():
    planner = scripted(
        "You can list them with:\n```bash\nkubectl get pods -n demo --field-selector status.phase!=Running\n```",
        '<tool_call>\n{"name": "run_kubectl", "arguments": {"command": "kubectl get pods -n demo --field-selector status.phase!=Running"}}',
        "No pods are failing.",
    )
    cluster = FakeCluster()
    events = list(Agent(planner, make_toolbox(cluster)).ask("which pods are not running in demo?"))
    assert [e.type for e in events] == ["tool_call", "tool_result", "final"]
    assert cluster.commands[0][0][:2] == ("get", "pods")


def _pending_web(cluster):
    planner = scripted(
        call("audit_manifest", kind="Deployment", name="web", namespace="demo"),
        call("apply_fix", kind="Deployment", name="web", namespace="demo"),
        "I prepared a fix.",
        "Done.",
    )
    agent = Agent(planner, make_toolbox(cluster))
    list(agent.ask("fix web"))
    return agent


def test_image_ops_name_the_container_and_the_suggested_tag():
    agent = _pending_web(FakeCluster())
    assert agent.toolbox.pending.image_ops == [
        {"index": 0, "path": "/spec/template/spec/containers/0/image", "value": "nginx:1.25", "container": "app"}
    ]


def test_a_chosen_image_replaces_the_models_tag_and_is_dry_run_again():
    cluster = FakeCluster()
    agent = _pending_web(cluster)
    events = list(agent.resume(approved=True, image_overrides={0: "nginx:1.27.2"}))
    applied = cluster.applied()[-1][1]
    assert "nginx:1.27.2" in applied and "nginx:1.25" not in applied
    dry_runs = [c for c in cluster.commands if "--dry-run=server" in c[0]]
    assert len(dry_runs) == 2 and "nginx:1.27.2" in dry_runs[-1][1]
    assert "app image set to nginx:1.27.2" in events[0].text


def test_an_unpinned_chosen_image_is_refused():
    cluster = FakeCluster()
    agent = _pending_web(cluster)
    events = list(agent.resume(approved=True, image_overrides={0: "nginx:latest"}))
    assert events[0].text.startswith("Not applied") and cluster.applied() == []


def test_running_images_reads_the_pods_digests():
    pods = {"items": [{"status": {"containerStatuses": [
        {"name": "app", "imageID": "docker-pullable://nginx@sha256:" + "a" * 64},
        {"name": "sidecar", "imageID": "docker://sha256:" + "b" * 64},  # local image: no usable digest
    ]}}]}

    def kubectl(*args, input_text=None, timeout=None):
        assert args[:2] == ("get", "pods") and "-l" in args
        return True, json.dumps(pods)

    toolbox = make_toolbox(FakeCluster())
    toolbox.kubectl = kubectl
    manifest = yaml.safe_dump({"kind": "Deployment", "spec": {"selector": {"matchLabels": {"app": "web"}}}})
    assert toolbox.running_images(manifest, "demo") == {"app": "nginx@sha256:" + "a" * 64}


def test_protected_namespaces_are_refused_and_hidden():
    cluster = FakeCluster()
    toolbox = make_toolbox(cluster)
    toolbox.protected_namespaces = frozenset({"keycloak"})
    toolbox.list_resources = lambda: (
        [{"kind": "Deployment", "namespace": "demo", "name": "web"}, {"kind": "Deployment", "namespace": "keycloak", "name": "keycloak"}],
        None,
    )
    assert "keycloak" not in toolbox.tool_list_resources()
    assert toolbox.tool_list_resources(namespace="keycloak").startswith("Refused")
    for result in (
        toolbox.tool_get_manifest("Deployment", "keycloak", "keycloak"),
        toolbox.tool_audit_manifest("Deployment", "keycloak", "keycloak"),
        toolbox.tool_apply_fix("Deployment", "keycloak", "keycloak"),
        toolbox.tool_run_kubectl("kubectl get pods -n keycloak"),
        toolbox.tool_run_kubectl("kubectl logs keycloak-0 --namespace=keycloak"),
    ):
        assert result.startswith("Refused"), result
    assert cluster.commands == []


def test_all_namespaces_output_hides_protected_rows():
    toolbox = make_toolbox(FakeCluster())
    toolbox.protected_namespaces = frozenset({"keycloak"})
    toolbox.kubectl = lambda *a, **k: (True, "NAMESPACE  NAME\ndemo  web-1\nkeycloak  keycloak-0\n")
    out = toolbox.tool_run_kubectl("kubectl get pods -A")
    assert "web-1" in out and "keycloak-0" not in out


def test_a_viewer_cannot_apply_a_fix():
    cluster = FakeCluster()
    toolbox = make_toolbox(cluster)
    toolbox.can_apply = False
    toolbox.tool_audit_manifest("Deployment", "web", "demo")
    result = toolbox.tool_apply_fix("Deployment", "web", "demo")
    assert result.startswith("Refused") and "viewer" in result
    assert toolbox.pending is None and cluster.applied() == []


def test_created_resources_are_labelled_for_reset():
    cluster = FakeCluster()
    with_secret = dict(AUDIT, new_resources=["apiVersion: v1\nkind: Secret\nmetadata:\n  name: web-secrets\nstringData:\n  pw: real"])
    toolbox = make_toolbox(cluster, with_secret)
    toolbox.created_label = ("app.kubernetes.io/managed-by", "cluster-agent")
    toolbox.tool_audit_manifest("Deployment", "web", "demo")
    toolbox.tool_apply_fix("Deployment", "web", "demo")
    toolbox.execute_pending()
    secret = yaml.safe_load(cluster.applied()[0][1])
    assert secret["metadata"]["labels"] == {"app.kubernetes.io/managed-by": "cluster-agent"}



def test_change_description_explains_without_leaking_the_old_value():
    cluster = FakeCluster()
    audit = {
        "findings": [{"rule_id": "KSEC-001", "severity": "critical", "doc": 0, "path": "/spec/template/spec/containers/0/env/0/value",
                      "message": "Plaintext credential: value under sensitive key 'DB_PASSWORD'", "evidence": "hunt***"}],
        "patch": [{"doc": 0, "op": "replace", "path": "/spec/template/spec/containers/0/env/0",
                   "value": {"name": "DB_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "app-secrets", "key": "db-password"}}}}],
        "new_resources": ["apiVersion: v1\nkind: Secret\nmetadata:\n  name: app-secrets\nstringData:\n  db-password: '<REPLACE_WITH_SECRET_VALUE>'"],
        "notes": [],
    }
    manifest = yaml.safe_dump({"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"},
                               "spec": {"template": {"spec": {"containers": [{"name": "app", "image": "app:1",
                                        "env": [{"name": "DB_PASSWORD", "value": "hunter2secret"}]}]}}}}, sort_keys=False)
    toolbox = make_toolbox(cluster, audit)
    toolbox.fetch_yaml = lambda kind, namespace, name: (manifest, None)
    toolbox.tool_audit_manifest("Deployment", "web", "demo")
    result = toolbox.tool_apply_fix("Deployment", "web", "demo")
    assert "DB_PASSWORD: its plaintext value is replaced by a reference to Secret 'app-secrets' (key 'db-password')" in result
    assert "create Secret 'app-secrets' in demo with key(s) db-password" in result
    assert "KSEC-001 (critical)" in result
    assert "hunter2secret" not in result
