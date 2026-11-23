"""The cluster tools of demo/agent.py's Toolbox (list, extract, get, validate,
apply YAML, logs, diagnostics, scale), against a fake kubectl."""

import json

import yaml

from demo.agent import MAX_LOG_LINES, MAX_REPLICAS, Agent, canonical_kind, mask_log_secrets, yaml_in_message
from demo.audit_log import build_change_record
from demo.test_agent import call, make_toolbox, scripted

DEPLOYMENT = {
    "apiVersion": "apps/v1", "kind": "Deployment",
    "metadata": {"name": "web", "namespace": "demo", "uid": "u-1", "managedFields": [{"manager": "kubectl"}]},
    "spec": {"replicas": 2, "selector": {"matchLabels": {"app": "web"}},
             "template": {"metadata": {"labels": {"app": "web"}}, "spec": {"containers": [{"name": "app", "image": "nginx:1.25"}]}}},
    "status": {"readyReplicas": 2},
}


def crashing_pod(name="web-7d4b9-abcde", containers=("app",)):
    return {
        "apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": "demo"},
        "spec": {"nodeName": "minikube", "containers": [{"name": c, "image": "nginx:1.25"} for c in containers]},
        "status": {
            "phase": "Running", "startTime": "2026-11-22T10:00:00Z",
            "conditions": [{"type": "Ready", "status": "False", "reason": "ContainersNotReady"}],
            "containerStatuses": [{
                "name": "app", "image": "nginx:1.25", "ready": False, "restartCount": 4,
                "state": {"waiting": {"reason": "CrashLoopBackOff", "message": "back-off 1m20s"}},
                "lastState": {"terminated": {"reason": "OOMKilled", "exitCode": 137, "finishedAt": "2026-11-22T10:05:00Z"}},
            }],
        },
    }


class FakeKube:
    """Answers the kubectl calls the tools make, from a few in-memory objects."""

    def __init__(self, objects=(), pods=(), events=(), logs="", dry_run_ok=True):
        self.objects = {(o["kind"].lower(), o["metadata"]["namespace"], o["metadata"]["name"]): o for o in objects}
        self.pods, self.events, self.logs, self.dry_run_ok = list(pods), list(events), logs, dry_run_ok
        self.commands = []

    def kubectl(self, *args, input_text=None, timeout=None):
        self.commands.append((args, input_text))
        if args[0] == "apply":
            if "--dry-run=server" in args:
                return (True, "configmap/cfg created (server dry run)") if self.dry_run_ok else (False, 'Deployment.apps "web" is invalid')
            return True, "configured"
        if args[0] == "logs":
            return True, self.logs
        if args[:2] == ("get", "pod") and len(args) > 2 and not args[2].startswith("-"):
            pod = next((p for p in self.pods if p["metadata"]["name"] == args[2]), None)
            return (True, json.dumps(pod)) if pod else (False, f'pods "{args[2]}" not found')
        if args[:2] == ("get", "pods"):
            namespace = args[args.index("-n") + 1] if "-n" in args else None
            return True, json.dumps({"items": [p for p in self.pods if namespace in (None, p["metadata"]["namespace"])]})
        if args[:2] == ("get", "events"):
            return True, json.dumps({"items": self.events})
        if args[0] == "get":
            kind = canonical_kind(args[1])
            items = [o for (k, _, _), o in self.objects.items() if k == kind]
            return True, json.dumps({"items": items})
        return False, f"unexpected kubectl call {args}"

    def fetch_yaml(self, kind, namespace, name):
        obj = self.objects.get((canonical_kind(kind), namespace, name))
        if obj is None:
            return None, f'{kind} "{name}" not found'
        return yaml.safe_dump({k: v for k, v in obj.items() if k != "status"}, sort_keys=False), None

    def applied(self):
        return [c for c in self.commands if c[0][0] == "apply" and "--dry-run=server" not in c[0]]


def toolbox_for(kube, **overrides):
    toolbox = make_toolbox(kube)
    toolbox.kubectl, toolbox.fetch_yaml = kube.kubectl, kube.fetch_yaml
    toolbox.protected_namespaces = frozenset({"keycloak", "audit"})
    toolbox.created_label = ("app.kubernetes.io/managed-by", "cluster-agent")
    for key, value in overrides.items():
        setattr(toolbox, key, value)
    return toolbox


# ------------------------------------------------------------- guardrails --


def test_names_that_look_like_flags_never_reach_kubectl():
    kube = FakeKube([DEPLOYMENT])
    toolbox = toolbox_for(kube)
    for result in (
        toolbox.tool_get_specific_object_yaml("Deployment", "--all-namespaces", "demo"),
        toolbox.tool_get_pod_logs("web", "--kubeconfig=/tmp/x"),
        toolbox.tool_list_k8s_resource_names("pods;rm", "demo"),
        toolbox.tool_scale_resource("Deployment", "Web", "demo", 3),
        toolbox.tool_extract_k8s_resource_yamls("pods", "demo", label_selector="--selector=x"),
    ):
        assert result.startswith("Invalid"), result
    assert kube.commands == []


def test_secret_contents_are_refused_everywhere():
    kube = FakeKube()
    toolbox = toolbox_for(kube)
    secret_yaml = "apiVersion: v1\nkind: Secret\nmetadata:\n  name: db\nstringData:\n  password: x"
    for result in (
        toolbox.tool_list_k8s_resource_names("secrets", "demo"),
        toolbox.tool_extract_k8s_resource_yamls("secret", "demo"),
        toolbox.tool_get_specific_object_yaml("Secret", "db", "demo"),
        toolbox.tool_apply_yaml_to_cluster(secret_yaml, "demo"),
    ):
        assert result.startswith("Refused: the agent never reads or writes Secret contents"), result
    assert kube.commands == []


def test_protected_namespaces_are_refused_by_every_new_tool():
    kube = FakeKube()
    toolbox = toolbox_for(kube)
    for result in (
        toolbox.tool_list_k8s_resource_names("services", "audit"),
        toolbox.tool_extract_k8s_resource_yamls("statefulsets", "audit"),
        toolbox.tool_get_pod_logs("postgres-0", "audit"),
        toolbox.tool_get_pod_diagnostics("keycloak", "keycloak"),
        toolbox.tool_scale_resource("StatefulSet", "postgres", "audit", 0),
        toolbox.tool_validate_k8s_yaml_dry_run("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: x\n  namespace: audit"),
        toolbox.tool_apply_yaml_to_cluster("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: x", "keycloak"),
    ):
        assert result.startswith("Refused"), result
    assert kube.commands == []


def test_canonical_kind_and_log_masking():
    assert [canonical_kind(k) for k in ("Deployment", "deployments", "deploy", "deployments.apps", "ingresses", "rc")] == [
        "deployment", "deployment", "deployment", "deployment", "ingress", "replicationcontroller"]
    masked = mask_log_secrets("connecting with DB_PASSWORD=hunter2secret user=app\napi_key: sk-12345678 ok")
    assert "hunter2secret" not in masked and "DB_PASSWORD=hunt***" in masked and "api_key: sk-1***" in masked
    assert "user=app" in masked


# ---------------------------------------------------------------- reading --


def test_list_names_of_a_type_hides_protected_namespaces():
    kube = FakeKube([
        {"kind": "Service", "metadata": {"name": "web", "namespace": "demo"}},
        {"kind": "Service", "metadata": {"name": "postgres", "namespace": "audit"}},
    ])
    out = toolbox_for(kube).tool_list_k8s_resource_names("services")
    assert out == "Service demo/web"
    assert kube.commands[0][0] == ("get", "services", "-A", "-o", "json")


def test_list_workloads_keeps_the_original_behaviour():
    out = toolbox_for(FakeKube()).tool_list_k8s_resource_names(namespace="demo")
    assert out == "Deployment demo/web"


def test_extract_strips_runtime_fields_and_caps_the_count():
    many = [dict(DEPLOYMENT, metadata=dict(DEPLOYMENT["metadata"], name=f"web-{i}")) for i in range(12)]
    out = toolbox_for(FakeKube(many)).tool_extract_k8s_resource_yamls("deployments", "demo", "app=web")
    header, body = out.split("\n", 1)
    assert header == "12 deployments in demo (the first 10 below):"
    docs = list(yaml.safe_load_all(body))
    assert len(docs) == 10 and "status" not in docs[0] and "managedFields" not in docs[0]["metadata"]


# ------------------------------------------------------------- validating --


def test_validate_sets_the_namespace_and_changes_nothing():
    kube = FakeKube()
    out = toolbox_for(kube).tool_validate_k8s_yaml_dry_run("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cfg\ndata:\n  a: b", "demo")
    assert out.startswith("Valid") and "nothing was created or changed" in out
    assert "would be created" in out and "created (server dry run)" not in out
    (args, sent), = kube.commands
    assert "--dry-run=server" in args and yaml.safe_load(sent)["metadata"]["namespace"] == "demo"
    assert kube.applied() == []


def test_validate_reports_bad_yaml_and_refuses_cluster_scoped_objects():
    toolbox = toolbox_for(FakeKube())
    assert toolbox.tool_validate_k8s_yaml_dry_run("kind: [unclosed").startswith("Not valid: Invalid YAML")
    assert toolbox.tool_validate_k8s_yaml_dry_run("kind: ConfigMap").startswith("Not valid: Every object needs")
    out = toolbox.tool_validate_k8s_yaml_dry_run("apiVersion: rbac.authorization.k8s.io/v1\nkind: ClusterRoleBinding\nmetadata:\n  name: x")
    assert out.startswith("Refused") and "cluster-scoped" in out
    assert toolbox_for(FakeKube(dry_run_ok=False)).tool_validate_k8s_yaml_dry_run(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: c").startswith("Not valid: the API server rejected it")


# --------------------------------------------------------------- applying --

NEW_CONFIGMAP = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: settings\ndata:\n  LOG_LEVEL: info\n"


def test_apply_yaml_waits_for_confirmation_then_creates_a_labelled_object():
    kube = FakeKube()
    toolbox = toolbox_for(kube)
    result = toolbox.tool_apply_yaml_to_cluster(NEW_CONFIGMAP, "demo")
    change = toolbox.pending
    assert change.action == "apply" and change.original_yaml == "" and kube.applied() == []
    assert "create ConfigMap demo/settings" in result
    assert yaml.safe_load(change.patched_yaml)["metadata"]["labels"] == {"app.kubernetes.io/managed-by": "cluster-agent"}
    assert toolbox.execute_pending().startswith("Applied")
    assert "settings" in kube.applied()[0][1]


def test_apply_yaml_updating_an_existing_object_is_not_labelled():
    kube = FakeKube([DEPLOYMENT])
    toolbox = toolbox_for(kube)
    updated = yaml.safe_dump({k: v for k, v in DEPLOYMENT.items() if k != "status"}).replace("nginx:1.25", "nginx:1.27")
    result = toolbox.tool_apply_yaml_to_cluster(updated)
    assert "update Deployment demo/web" in result and "2 line(s) differ" in result
    assert "labels" not in yaml.safe_load(toolbox.pending.patched_yaml)["metadata"]


def test_apply_yaml_refusals():
    kube = FakeKube()
    toolbox = toolbox_for(kube)
    assert toolbox.tool_apply_yaml_to_cluster(NEW_CONFIGMAP + "---\n" + NEW_CONFIGMAP.replace("settings", "b")).startswith("Send one object per call")
    viewer = toolbox_for(kube, can_apply=False)
    assert "viewer" in viewer.tool_apply_yaml_to_cluster(NEW_CONFIGMAP)
    assert toolbox_for(FakeKube(dry_run_ok=False)).tool_apply_yaml_to_cluster(NEW_CONFIGMAP).startswith("The YAML failed")
    assert kube.applied() == [] and toolbox.pending is None


def test_only_one_change_waits_at_a_time():
    toolbox = toolbox_for(FakeKube([DEPLOYMENT]))
    toolbox.tool_apply_yaml_to_cluster(NEW_CONFIGMAP, "demo")
    assert "already waiting for the user's confirmation" in toolbox.tool_scale_resource("Deployment", "web", "demo", 3)


# ---------------------------------------------------------------- scaling --


def test_scale_waits_for_confirmation_then_applies_the_new_replica_count():
    kube = FakeKube([DEPLOYMENT])
    toolbox = toolbox_for(kube)
    result = toolbox.tool_scale_resource("deployments", "web", "demo", "3")
    change = toolbox.pending
    assert "scale deployment demo/web from 2 to 3 replica(s)" in result
    assert change.action == "scale" and change.patch == [{"doc": 0, "op": "replace", "path": "/spec/replicas", "value": 3}]
    assert kube.applied() == []
    toolbox.execute_pending()
    assert yaml.safe_load(kube.applied()[0][1])["spec"]["replicas"] == 3


def test_scale_refusals():
    toolbox = toolbox_for(FakeKube([DEPLOYMENT]))
    assert "can't be scaled" in toolbox.tool_scale_resource("DaemonSet", "web", "demo", 2)
    assert toolbox.tool_scale_resource("Deployment", "web", "demo", MAX_REPLICAS + 1).startswith("Refused")
    assert toolbox.tool_scale_resource("Deployment", "web", "demo", "many").startswith("replicas must be a whole number")
    assert "already has 2 replica(s)" in toolbox.tool_scale_resource("Deployment", "web", "demo", 2)
    assert toolbox.pending is None


def test_a_scale_change_is_recorded_as_such_in_the_audit_log():
    toolbox = toolbox_for(FakeKube([DEPLOYMENT]))
    toolbox.tool_scale_resource("Deployment", "web", "demo", 0)
    record = build_change_record(toolbox.pending, "applied", "Applied")
    assert record["action"] == "scale" and record["findings"] == []
    assert record["operations"][0]["value"] == 0 and "-  replicas: 2\n+  replicas: 0" in record["diff"]


def test_a_requested_fix_is_not_forced_after_another_change_tool_this_turn():
    kube = FakeKube([DEPLOYMENT])
    toolbox = toolbox_for(kube)
    toolbox.tool_audit_manifest("Deployment", "web", "demo")  # an earlier audit with a fix ready
    planner = scripted(call("apply_yaml_to_cluster", manifest=NEW_CONFIGMAP, namespace="audit"), "That namespace is protected.")
    events = list(Agent(planner, toolbox).ask("apply this config map"))
    assert [e.type for e in events] == ["tool_call", "tool_result", "final"] and toolbox.pending is None


# ----------------------------------------------------------------- pods --


def test_logs_fall_back_to_the_workloads_pods_and_mask_secrets():
    kube = FakeKube(pods=[crashing_pod(), crashing_pod("web-7d4b9-fghij")], logs="started\nDB_PASSWORD=hunter2secret\n")
    out = toolbox_for(kube).tool_get_pod_logs("web", "demo", tail_lines=5000, previous="true")
    assert "'web' is a workload; its pod(s): demo/web-7d4b9-abcde, demo/web-7d4b9-fghij. Showing web-7d4b9-abcde." in out
    assert "hunter2secret" not in out and "DB_PASSWORD=hunt***" in out
    args = kube.commands[-1][0]
    assert args[:2] == ("logs", "web-7d4b9-abcde") and f"--tail={MAX_LOG_LINES}" in args and "--previous" in args


def test_logs_of_a_multi_container_pod_cover_every_container():
    kube = FakeKube(pods=[crashing_pod(containers=("app", "sidecar"))], logs="")
    out = toolbox_for(kube).tool_get_pod_logs("web-7d4b9-abcde", "demo")
    assert "--all-containers=true" in kube.commands[-1][0] and out.endswith("(no log output)")


def test_unknown_pod():
    assert toolbox_for(FakeKube()).tool_get_pod_logs("nope", "demo") == "No pod or workload called 'nope' in namespace 'demo'."


def test_without_a_namespace_pods_are_found_in_any_visible_namespace():
    kube = FakeKube(pods=[crashing_pod()], logs="hello")
    out = toolbox_for(kube).tool_get_pod_logs("web")
    assert "demo/web-7d4b9-abcde" in out and kube.commands[-1][0][:4] == ("logs", "web-7d4b9-abcde", "-n", "demo")
    assert kube.commands[0][0] == ("get", "pods", "-A", "-o", "json")
    assert "Pod demo/web-7d4b9-abcde" in toolbox_for(FakeKube(pods=[crashing_pod()])).tool_get_pod_diagnostics("web")


def test_diagnostics_explain_a_crashing_pod():
    events = [
        {"type": "Warning", "reason": "BackOff", "message": "Back-off restarting failed container", "count": 9, "lastTimestamp": "2026-11-22T10:06:00Z"},
        {"type": "Normal", "reason": "Pulled", "message": "Container image already present", "count": 5, "lastTimestamp": "2026-11-22T10:01:00Z"},
    ]
    out = toolbox_for(FakeKube(pods=[crashing_pod()], events=events)).tool_get_pod_diagnostics("web-7d4b9-abcde", "demo")
    assert "Pod demo/web-7d4b9-abcde: Running, node minikube" in out
    assert "condition Ready is False: ContainersNotReady" in out
    assert "restarts=4, waiting: CrashLoopBackOff -- back-off 1m20s" not in out  # message follows the reason after a space
    assert "waiting: CrashLoopBackOff back-off 1m20s" in out
    assert "last run: OOMKilled (exit code 137)" in out
    assert out.index("Pulled") < out.index("BackOff (x9)")  # oldest first
    assert "get_pod_logs with previous=true" in out and "memory limit" in out


# ----------------------------------------------------- credential masking --

LEAKY = dict(DEPLOYMENT, spec=dict(DEPLOYMENT["spec"], template={"metadata": {"labels": {"app": "web"}}, "spec": {"containers": [
    {"name": "app", "image": "nginx:1.25", "env": [{"name": "DB_PASSWORD", "value": "hunter2-Zq8vT3xK"}, {"name": "MODE", "value": "prod"}]}]}}))


def test_read_tools_mask_plaintext_credentials():
    toolbox = toolbox_for(FakeKube([LEAKY]))
    for out in (toolbox.tool_get_specific_object_yaml("Deployment", "web", "demo"), toolbox.tool_extract_k8s_resource_yamls("deployments", "demo")):
        assert "hunter2-Zq8vT3xK" not in out and "value: hunt***" in out and "value: prod" in out
        assert out.endswith("the real values are never shown.")
    assert not toolbox_for(FakeKube([DEPLOYMENT])).tool_get_specific_object_yaml("Deployment", "web", "demo").endswith("never shown.")


def test_the_auditor_still_sees_the_real_manifest():
    seen = []
    toolbox = toolbox_for(FakeKube([LEAKY]))
    toolbox.audit = lambda text: (seen.append(text), ({"findings": [], "patch": [], "new_resources": [], "notes": []}, []))[1]
    toolbox.tool_audit_manifest("Deployment", "web", "demo")
    assert "hunter2-Zq8vT3xK" in seen[0]


def test_yaml_with_masked_values_is_never_applied():
    kube = FakeKube([LEAKY])
    toolbox = toolbox_for(kube)
    copied = toolbox.tool_get_specific_object_yaml("Deployment", "web", "demo").replace("replicas: 2", "replicas: 3")
    out = toolbox.tool_apply_yaml_to_cluster(copied)
    assert out.startswith("Refused: the YAML contains masked values") and toolbox.pending is None
    assert not any(c[0][0] == "apply" for c in kube.commands)


def test_diagnostics_explain_a_command_that_cannot_start():
    pod = crashing_pod()
    pod["status"]["containerStatuses"][0]["lastState"] = {"terminated": {"reason": "ContainerCannotRun", "exitCode": 127}}
    out = toolbox_for(FakeKube(pods=[pod])).tool_get_pod_diagnostics("web-7d4b9-abcde", "demo")
    hints = out.split("hints:")[1]
    assert hints.index("exit code 127") < hints.index("previous=true")


def test_a_check_request_answered_from_the_yaml_is_sent_to_the_auditor():
    # Seen on the live run: the planner read the YAML and reviewed it itself.
    kube = FakeKube([DEPLOYMENT])
    toolbox = toolbox_for(kube)
    planner = scripted(
        call("get_specific_object_yaml", kind="Deployment", name="web", namespace="demo"),
        "The deployment has a plaintext password and no resource limits.",  # its own review: not trusted
        "The audit found one problem; I prepared the fix.",
    )
    events = list(Agent(planner, toolbox).ask("check the web deployment in demo and fix it"))
    calls = [(e.name, e.arguments) for e in events if e.type == "tool_call"]
    assert calls[1] == ("audit_manifest", {"kind": "Deployment", "name": "web", "namespace": "demo"})
    assert calls[2][0] == "apply_fix" and toolbox.pending is not None


def test_a_plain_read_request_is_not_sent_to_the_auditor():
    planner = scripted(call("get_specific_object_yaml", kind="Deployment", name="web", namespace="demo"), "Here it is.")
    events = list(Agent(planner, toolbox_for(FakeKube([DEPLOYMENT]))).ask("show me the web deployment in demo"))
    assert [e.name for e in events if e.type == "tool_call"] == ["get_specific_object_yaml"]


def test_change_tools_are_refused_when_the_user_did_not_ask_for_a_change():
    # Seen on the live run: while diagnosing a crash, the planner tried a scale.
    kube = FakeKube([DEPLOYMENT], pods=[crashing_pod()])
    toolbox = toolbox_for(kube)
    planner = scripted(
        call("get_pod_diagnostics", pod_name="web", namespace="demo"),
        call("scale_resource", kind="Deployment", name="web", namespace="demo", replicas=3),
        "It runs out of memory.",
    )
    events = list(Agent(planner, toolbox).ask("why is web in demo crashing?"))
    assert events[3].text.startswith("Not done: the user didn't ask for a change")
    assert toolbox.pending is None and not any(c[0][0] == "apply" for c in kube.commands)


def test_yaml_to_create_is_proposed_when_the_planner_only_answers_in_text():
    # Seen on the live run: the planner asked the user to confirm, in text.
    kube = FakeKube()
    toolbox = toolbox_for(kube)
    planner = scripted("Please confirm if this YAML should be applied to your cluster.", "I prepared it; confirm to create it.")
    events = list(Agent(planner, toolbox).ask("create this ConfigMap in the demo namespace:\n" + NEW_CONFIGMAP))
    assert [e.type for e in events] == ["tool_call", "confirm", "final"]
    assert events[0].arguments["namespace"] == "demo" and toolbox.pending.namespace == "demo"


def test_yaml_in_message():
    manifest, rest = yaml_in_message("please apply:\n```yaml\n" + NEW_CONFIGMAP + "```\nthanks")
    assert manifest == NEW_CONFIGMAP and "apply" in rest and "ConfigMap" not in rest
    assert yaml_in_message("kind is not yaml here")[0] is None
    assert yaml_in_message("create a configmap called settings")[0] is None
