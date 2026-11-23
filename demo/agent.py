"""
A basic tool-using agent for the cluster, built on the models the demo
already loads -- no extra model, no extra training:

  - the PLANNER is the base Qwen2.5-Coder-7B-Instruct (the fine-tuned
    adapter temporarily disabled, like the demo's chat mode). Its own chat
    template supports native tool calling: it answers with
    <tool_call>{"name": ..., "arguments": ...}</tool_call> blocks;
  - the AUDITOR is the same model with the fine-tuned adapter enabled,
    exposed to the planner as the `audit_manifest` tool.

Loop: the planner picks a tool, the tool runs, its result goes back into the
conversation, until the planner answers in plain text (or MAX_STEPS is hit).

Tools: list_k8s_resource_names, extract_k8s_resource_yamls,
get_specific_object_yaml, get_pod_logs, get_pod_diagnostics and
validate_k8s_yaml_dry_run read; audit_manifest runs the auditor; apply_fix,
scale_resource and apply_yaml_to_cluster change the cluster; run_kubectl is
a read-only fallback.

Guardrails:
  - read-only tools run freely; `run_kubectl` only accepts read-only verbs;
    no tool reads or writes Secret contents, and credential-looking values
    in pod logs are masked;
  - every name, namespace and type the model passes is validated before it
    becomes a kubectl argument, and protected namespaces are refused;
  - the only ways to change the cluster are apply_fix, scale_resource and
    apply_yaml_to_cluster. Each validates the change with a server-side dry
    run and then PAUSES the agent until the user confirms or declines it in
    the UI (Agent.resume); only one change waits at a time.

The planner and the tools are injected (Agent(planner, toolbox)), so the
loop is tested with a scripted fake planner and a fake kubectl -- see
demo/test_agent.py.
"""

from __future__ import annotations

import difflib
import json
import re
import shlex
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

MAX_STEPS = 8
MAX_TOOL_RESULT_CHARS = 2500
MAX_EXTRACTED_OBJECTS = 10
MAX_LOG_LINES = 200
MAX_REPLICAS = 10  # a demo guardrail: the planner can't scale anything to hundreds of pods
WORKLOADS = "workloads"
# The tools that change the cluster. Each one only prepares a change, which
# waits for the user's Confirm/Decline (and is recorded in the audit log).
MUTATING_TOOLS = {"apply_fix", "apply_yaml_to_cluster", "scale_resource"}
SCALABLE_KINDS = {"deployment", "statefulset", "replicaset", "replicationcontroller"}
# The agent only works on namespaced objects (its RBAC can't change the
# others anyway).
CLUSTER_SCOPED_KINDS = {
    "namespace", "node", "persistentvolume", "clusterrole", "clusterrolebinding", "customresourcedefinition", "storageclass",
    "priorityclass", "ingressclass", "runtimeclass", "apiservice", "mutatingwebhookconfiguration", "validatingwebhookconfiguration",
}

PLANNER_SYSTEM_PROMPT = """\
You are a Kubernetes cluster agent. You help the user inspect and troubleshoot their cluster, and find \
and fix security and configuration problems in its workloads.

How to work:
- Use the tools to get facts. Never invent cluster state, logs or audit results: only report what the \
tools return.
- To see what exists, call list_k8s_resource_names (resource_type "workloads" lists every workload). To \
read objects, call get_specific_object_yaml (one object) or extract_k8s_resource_yamls (every object of a \
type in a namespace).
- To find out why a pod fails, call get_pod_diagnostics first, then get_pod_logs (previous=true shows the \
run before the last crash).
- To check a workload for security and configuration problems, ALWAYS call audit_manifest -- never \
review the YAML yourself. It runs a specialized auditing model and returns the problems it found and \
whether a fix is ready. To fix them, \
call apply_fix -- right after the audit, if the user asked you to fix something. After a fix is applied, \
call audit_manifest again to verify it.
- To change the number of replicas, call scale_resource. To create or update an object from YAML, call \
apply_yaml_to_cluster; to only check YAML without changing anything, call validate_k8s_yaml_dry_run.
- apply_fix, scale_resource and apply_yaml_to_cluster never change anything by themselves: the change is \
validated with a server-side dry run and shown to the user, who confirms or declines it -- do not ask for \
confirmation yourself.
- For any other read-only question, use run_kubectl.
- When you need a tool, call it right away, in this same answer. Never reply with "please wait" or describe a call without making it. A call looks like this:
<tool_call>
{"name": "audit_manifest", "arguments": {"kind": "Deployment", "name": "web", "namespace": "demo"}}
</tool_call>
- If the user is just chatting, answer without tools.
- Keep answers short and factual."""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


_KIND = {"type": "string", "description": "e.g. Deployment"}
_NAME = {"type": "string"}
_NAMESPACE = {"type": "string", "description": "Defaults to 'default'."}
_POD = {"type": "string", "description": "A pod name, or a workload name to use its pods."}
_POD_NAMESPACE = {"type": "string", "description": "Omit to search every namespace."}

TOOLS = [
    _tool("list_k8s_resource_names", "List the names of the resources of one type. resource_type 'workloads' (the default) lists "
          "every workload (pods, deployments, statefulsets, daemonsets, replicasets, jobs, cronjobs).",
          {"resource_type": {"type": "string", "description": "'workloads', or a type such as services, pods, configmaps."},
           "namespace": {"type": "string", "description": "Only this namespace. Omit for all namespaces."}}, []),
    _tool("extract_k8s_resource_yamls", "Get the YAML of every resource of one type in a namespace (at most "
          f"{MAX_EXTRACTED_OBJECTS} objects), optionally filtered by a label selector.",
          {"resource_type": {"type": "string", "description": "e.g. deployments, services"}, "namespace": _NAMESPACE,
           "label_selector": {"type": "string", "description": "e.g. app=web"}}, ["resource_type"]),
    _tool("get_specific_object_yaml", "Get the YAML of one resource.", {"kind": _KIND, "name": _NAME, "namespace": _NAMESPACE}, ["kind", "name"]),
    _tool("audit_manifest", "Check one resource for security and configuration problems (plaintext credentials, unpinned images, "
          "privileged containers, host access, broken selectors, probe and resource mistakes, misspelled names...). "
          "Returns the problems found and whether a fix is ready.", {"kind": _KIND, "name": _NAME, "namespace": _NAMESPACE}, ["kind", "name"]),
    _tool("apply_fix", "Apply the fix from the latest audit of a resource. It is validated with a server-side dry run, "
          "then the user is asked to confirm before anything changes.", {"kind": _KIND, "name": _NAME, "namespace": _NAMESPACE}, ["kind", "name"]),
    _tool("validate_k8s_yaml_dry_run", "Check whether YAML would be accepted by the cluster (server-side dry run). Nothing is changed.",
          {"manifest": {"type": "string", "description": "The YAML text."},
           "namespace": {"type": "string", "description": "For objects without metadata.namespace. Defaults to 'default'."}}, ["manifest"]),
    _tool("apply_yaml_to_cluster", "Create or update one resource from YAML. It is validated with a server-side dry run, then the "
          "user is asked to confirm before anything changes.",
          {"manifest": {"type": "string", "description": "The YAML of ONE object."},
           "namespace": {"type": "string", "description": "If the YAML has no metadata.namespace. Defaults to 'default'."}}, ["manifest"]),
    _tool("get_pod_logs", "Get the latest log lines of a pod.",
          {"pod_name": _POD, "namespace": _POD_NAMESPACE, "container": {"type": "string", "description": "Omit for every container."},
           "tail_lines": {"type": "integer", "description": "Defaults to 100 (at most 200)."},
           "previous": {"type": "boolean", "description": "true: the logs of the run before the last restart (after a crash)."}},
          ["pod_name"]),
    _tool("get_pod_diagnostics", "Explain a pod's health: phase, container states, restarts, last termination reasons and recent "
          "events, with hints on the usual causes.", {"pod_name": _POD, "namespace": _POD_NAMESPACE}, ["pod_name"]),
    _tool("scale_resource", "Change the number of replicas of a Deployment, StatefulSet, ReplicaSet or ReplicationController. "
          "It is validated with a server-side dry run, then the user is asked to confirm.",
          {"kind": _KIND, "name": _NAME, "namespace": _NAMESPACE, "replicas": {"type": "integer"}}, ["kind", "name", "replicas"]),
    _tool("run_kubectl", "Run a READ-ONLY kubectl command (get, describe, logs, top, events, explain, version, ...). "
          "Commands that change the cluster are refused.",
          {"command": {"type": "string", "description": "e.g. kubectl get pods -n demo"}}, ["command"]),
]

READ_ONLY_VERBS = {"get", "describe", "logs", "top", "events", "explain", "api-resources", "api-versions", "version", "cluster-info"}
_SECRET_RESOURCES = {"secret", "secrets", "secret/"}
# The closing tag is optional at the very end: with a pre-filled
# "<tool_call>" the model often writes the call and stops without closing it.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL)


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def _fenced_tool_calls(text: str) -> tuple[list[str], str]:
    """Qwen2.5-Coder sometimes writes a call as a ```json block (or a bare
    JSON object) instead of its native <tool_call> tags -- seen on the very
    first end-to-end run. Accepted only when the JSON names a known tool."""
    blocks, kept = [], text
    candidates = _FENCED_JSON_RE.findall(text) or ([text.strip()] if text.strip().startswith("{") else [])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name") in _TOOL_NAMES and "arguments" in obj:
            blocks.append(candidate)
            kept = kept.replace(candidate, "")
    kept = re.sub(r"```(?:json)?\s*```", "", kept)
    return blocks, kept


def parse_tool_calls(text: str) -> tuple[list[dict], str, list[str]]:
    """(calls, remaining text, errors) from a planner answer, in Qwen's
    native <tool_call> format or as a JSON block naming a known tool. Each
    call is {"name": str, "arguments": dict}."""
    calls, errors = [], []
    blocks = _TOOL_CALL_RE.findall(text)
    remaining = _TOOL_CALL_RE.sub("", text).strip()
    if not blocks:
        blocks, remaining = _fenced_tool_calls(text)
        remaining = remaining.strip()
    for block in blocks:
        try:
            call = json.loads(block)
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            if not isinstance(call.get("name"), str) or not isinstance(arguments, dict):
                raise ValueError("needs a string 'name' and an object 'arguments'")
            calls.append({"name": call["name"], "arguments": arguments})
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            errors.append(f"could not parse tool call {block[:120]!r}: {e}")
    return calls, remaining, errors


def check_read_only(command: str) -> tuple[list[str] | None, str | None]:
    """(kubectl args, None) if `command` is a read-only kubectl command that
    may run, else (None, reason)."""
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return None, f"could not parse the command: {e}"
    if argv[:1] == ["kubectl"]:
        argv = argv[1:]
    if not argv:
        return None, "empty command"
    verb = argv[0]
    if verb not in READ_ONLY_VERBS:
        return None, f"'{verb}' can change the cluster or isn't allowed; only read-only commands run ({', '.join(sorted(READ_ONLY_VERBS))})"
    targets = [a.lower() for a in argv[1:] if not a.startswith("-")]
    reads_secret = any(t in _SECRET_RESOURCES or t.startswith("secret/") or t.startswith("secrets/") for t in targets)
    prints_content = any(a.startswith(("-o", "--output")) for a in argv) or verb == "describe"
    if verb in ("get", "describe") and reads_secret and prints_content:
        return None, "printing Secret contents is not allowed"
    return argv, None


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters truncated)"


# Arguments come from the model and become kubectl arguments: a name like
# "--all-namespaces" must never be taken for a flag.
_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")
_RESOURCE_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]{0,62}$")
_SELECTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/=,!() -]{0,250}$")
_KIND_ALIASES = {"deploy": "deployment", "sts": "statefulset", "rs": "replicaset", "rc": "replicationcontroller", "ds": "daemonset",
                 "po": "pod", "svc": "service", "cm": "configmap", "pvc": "persistentvolumeclaim", "ns": "namespace", "ing": "ingress"}
_KNOWN_KINDS = SCALABLE_KINDS | CLUSTER_SCOPED_KINDS | {
    "pod", "daemonset", "job", "cronjob", "service", "configmap", "secret", "ingress", "persistentvolumeclaim", "event", "endpoints",
    "serviceaccount", "role", "rolebinding", "networkpolicy", "horizontalpodautoscaler",
}
SECRETS_REFUSED = "Refused: the agent never reads or writes Secret contents (a credential fix creates its Secret through apply_fix)."
# key=value / key: value pairs whose key names a credential, in log lines.
_LOG_SECRET_RE = re.compile(r"(?i)\b([\w.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)[\w.-]*)(\s*[=:]\s*)([^\s,;\"']+)")


def canonical_kind(kind: str) -> str:
    """'Deployment', 'deployments', 'deploy', 'deployments.apps' -> 'deployment'."""
    k = _KIND_ALIASES.get(kind.lower().split(".")[0], kind.lower().split(".")[0])
    for candidate in (k, k[:-2] if k.endswith("es") else None, k[:-1] if k.endswith("s") else None):
        if candidate in _KNOWN_KINDS:
            return candidate
    return k


def mask_log_secrets(text: str) -> str:
    """Masks the values of credential-looking key=value pairs in log output
    (first 4 characters + "***", like the auditor's evidence)."""
    return _LOG_SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)[:4]}***", text)


def _where(item: dict) -> str:
    meta = item.get("metadata") or {}
    return f"{meta['namespace']}/{meta.get('name')}" if meta.get("namespace") else str(meta.get("name"))


MASKED_NOTE = "Note: plaintext credential values are masked (first 4 characters + ***); the real values are never shown."
_MASKED_VALUE_RE = re.compile(r"^.{0,4}\*\*\*$", re.DOTALL)


def redact_credentials(manifest_yaml: str) -> str:
    """The manifest with every plaintext credential the project's detectors
    find masked, plus a note when anything was -- what the read tools show,
    so a leaked credential isn't repeated in the chat. Unparseable YAML is
    returned as it is."""
    from demo.audit_log import redact_manifest

    try:
        redacted = redact_manifest(manifest_yaml)
    except Exception:  # noqa: BLE001 -- odd YAML: show it as kubectl returned it
        return manifest_yaml
    import yaml

    changed = list(yaml.safe_load_all(redacted)) != list(yaml.safe_load_all(manifest_yaml))
    return redacted + (MASKED_NOTE if changed else "")


def _has_masked_value(node) -> bool:
    if isinstance(node, dict):
        return any(_has_masked_value(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_masked_value(v) for v in node)
    return isinstance(node, str) and bool(_MASKED_VALUE_RE.match(node))


def _as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------


@dataclass
class PendingChange:
    """A validated change waiting for the user's confirmation."""

    kind: str
    name: str
    namespace: str
    findings: list[dict]
    original_yaml: str
    patched_yaml: str
    new_resources: list[str]
    dry_run_output: str
    patch: list[dict] = field(default_factory=list)
    running_images: dict = field(default_factory=dict)  # container name -> image@sha256 digest running now
    request: str = ""  # the user's message that led to the change (for the audit log)
    proposed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # "fix" (apply_fix), "apply" (apply_yaml_to_cluster) or "scale"
    # (scale_resource); for the last two, `summary` says what changes.
    action: str = "fix"
    summary: str = ""
    change_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        """JSON-safe, for the MCP tool layer (demo/mcp_server.py)."""
        return {**asdict(self), "proposed_at": self.proposed_at.isoformat()}

    @classmethod
    def from_dict(cls, data: dict) -> "PendingChange":
        return cls(**{**data, "proposed_at": datetime.fromisoformat(data["proposed_at"])})

    @property
    def diff(self) -> str:
        return unified_diff(self.original_yaml, self.patched_yaml)

    def describe(self) -> str:
        """The change in plain terms, for the planner -- so it can answer the
        user's questions about it. Never repeats a plaintext value that's
        being removed (it may be a credential)."""
        import yaml

        if self.action != "fix":
            return f"Proposed change: {self.summary}."
        doc = yaml.safe_load(self.original_yaml) or {}
        lines = []
        for op in self.patch:
            path, value = str(op.get("path", "")), op.get("value")
            old = _resolve_pointer(doc, path)
            if path.endswith("/image") and isinstance(value, str):
                container = (_resolve_pointer(doc, path.rsplit("/", 1)[0]) or {}).get("name", "?")
                lines.append(f"container '{container}' image: {old} -> {value} (a suggested tag; the user can set another before confirming)")
            elif "/env/" in path and isinstance(value, dict) and isinstance(value.get("valueFrom"), dict):
                ref = value["valueFrom"].get("secretKeyRef") or {}
                lines.append(
                    f"env var {value.get('name')}: its plaintext value is replaced by a reference to Secret "
                    f"'{ref.get('name')}' (key '{ref.get('key')}')"
                )
            elif op.get("op") == "remove":
                shown = "" if "/env/" in path or "/args/" in path or "/command/" in path else f" (was {json.dumps(old)[:80]})"
                lines.append(f"remove {path}{shown}")
            elif op.get("op") == "add":
                lines.append(f"add {path} = {json.dumps(value)[:120]}")
            else:
                shown = "" if "/env/" in path else f"{json.dumps(old)[:60]} -> "
                lines.append(f"set {path}: {shown}{json.dumps(value)[:80]}")
        for res in self.new_resources:
            try:
                created = yaml.safe_load(res) or {}
            except yaml.YAMLError:
                created = {}
            keys = sorted({**(created.get("stringData") or {}), **(created.get("data") or {})})
            lines.append(
                f"create {created.get('kind', 'resource')} '{(created.get('metadata') or {}).get('name')}' in {self.namespace}"
                + (f" with key(s) {', '.join(keys)} -- the user types the real value(s) before confirming" if keys else "")
            )
        problems = [f"{f['rule_id']} ({f['severity']}): {f['message']}" for f in self.findings]
        return (
            f"Proposed change to {self.kind} {self.namespace}/{self.name}:\n- " + "\n- ".join(lines)
            + "\nProblems it fixes:\n- " + "\n- ".join(problems)
        )

    @property
    def image_ops(self) -> list[dict]:
        """The patch's image changes, each with the container it affects --
        the values a user may want to choose instead of the model's guessed
        tag (the model can't know which versions exist in the registry)."""
        import yaml

        doc = yaml.safe_load(self.original_yaml) or {}
        out = []
        for index, op in enumerate(self.patch):
            if op.get("op") in ("replace", "add") and str(op.get("path", "")).endswith("/image") and isinstance(op.get("value"), str):
                container = _resolve_pointer(doc, op["path"].rsplit("/", 1)[0]) or {}
                out.append({"index": index, "path": op["path"], "value": op["value"], "container": container.get("name", "?")})
        return out


def _namespaces_in(argv: list[str]) -> list[str]:
    """The namespaces a kubectl argument list targets (-n X, --namespace X,
    --namespace=X)."""
    found = []
    for i, arg in enumerate(argv):
        if arg in ("-n", "--namespace") and i + 1 < len(argv):
            found.append(argv[i + 1])
        elif arg.startswith("--namespace="):
            found.append(arg.split("=", 1)[1])
        elif arg.startswith("-n") and len(arg) > 2 and not arg.startswith("--"):
            found.append(arg[2:])
    return found


def _with_label(resource_yaml: str, label: tuple | None) -> str:
    if label is None:
        return resource_yaml
    import yaml

    doc = yaml.safe_load(resource_yaml)
    if not isinstance(doc, dict):
        return resource_yaml
    doc.setdefault("metadata", {}).setdefault("labels", {})[label[0]] = label[1]
    return yaml.safe_dump(doc, sort_keys=False)


def unified_diff(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="before", tofile="after", lineterm=""))


def _resolve_pointer(doc, pointer: str):
    node = doc
    for token in pointer.strip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        elif isinstance(node, dict) and token in node:
            node = node[token]
        else:
            return None
    return node


def is_pinned(image: str) -> bool:
    from dataset.k8s import is_unpinned_image

    return bool(image.strip()) and not is_unpinned_image(image.strip())


@dataclass
class Toolbox:
    """The agent's tools. `audit` runs the fine-tuned auditor on a manifest
    and returns (parsed response or None, errors); `kubectl` runs a kubectl
    command and returns (ok, output) -- both injected so tests need neither
    a GPU nor a cluster."""

    audit: Callable[[str], tuple[dict | None, list[str]]]
    kubectl: Callable[..., tuple[bool, str]]
    list_resources: Callable[[], tuple[list[dict], str | None]]
    fetch_yaml: Callable[[str, str, str], tuple[str | None, str | None]]
    apply_patch: Callable[[str, list[dict]], tuple[str | None, str | None]]
    audits: dict = field(default_factory=dict)
    pending: PendingChange | None = None
    # Namespaces the agent must never read into or change (the login system,
    # its own identity) -- enforced here AND by the cluster's RBAC.
    protected_namespaces: frozenset = frozenset()
    # Whether the logged-in user may change the cluster (Keycloak role
    # "operator"); viewers can list, read and audit only.
    can_apply: bool = True
    # Label put on every resource the agent creates, so a reset can find them.
    created_label: tuple | None = None
    # The audit trail (demo/audit_log.AuditLog, or anything with ping() and
    # record(actor, adapter, record)): every decided change is recorded, and
    # none is applied while it's unreachable. None = auditing off.
    audit_log: Any = None
    actor: Any = None  # demo/audit_log.Actor of the logged-in user
    adapter: str | None = None  # the fine-tuned adapter doing the audits
    last_request: str = ""

    def _protected(self, namespace: str | None) -> str | None:
        if namespace in self.protected_namespaces:
            return f"Refused: the '{namespace}' namespace is protected; the agent doesn't read or change it."
        return None

    @staticmethod
    def _invalid(**values) -> str | None:
        """An error for the first argument that isn't a valid Kubernetes
        name (or resource type for kind/resource_type); None if all are."""
        for label, value in values.items():
            if value is None:
                continue
            is_type = label in ("kind", "resource_type")
            if not isinstance(value, str) or not (_RESOURCE_TYPE_RE if is_type else _NAME_RE).match(value):
                expected = "a resource type such as Deployment" if is_type else "a Kubernetes name (lowercase letters, digits, '-' and '.')"
                return f"Invalid {label} {value!r}: expected {expected}."
        return None

    def ready_fixes(self) -> list[tuple[str, str, str]]:
        """(kind, name, namespace) of the audits with a fix ready, oldest first."""
        return [key for key, a in self.audits.items() if a["result"]["patch"]]

    def set_session(self, can_apply: bool, actor=None, adapter: str | None = None, audit_log=None) -> None:
        """Who is using the agent, refreshed on every request (the role comes
        from the login, never from the chat)."""
        self.can_apply, self.actor, self.adapter, self.audit_log = can_apply, actor, adapter, audit_log

    def _change_refused(self, namespace: str) -> str | None:
        """Why a change can't even be prepared, or None."""
        if (refused := self._protected(namespace)):
            return refused
        if self.pending is not None:
            return (
                f"A change to {self.pending.kind} {self.pending.namespace}/{self.pending.name} is already waiting for the "
                "user's confirmation. Answer the user's questions about it; don't propose another change until they "
                "confirm or decline it."
            )
        if not self.can_apply:
            return (
                "Refused: the logged-in user has the 'viewer' role and can't change the cluster; only operators can. "
                "Tell the user the change is possible but needs an operator."
            )
        return None

    def _propose(self, change: PendingChange) -> str:
        self.pending = change
        return (
            "The change passed the server-side dry run and is waiting for the user's confirmation (the user clicks Confirm or "
            "Decline). Briefly tell the user what will change. Answer their questions about it; nothing is applied until they "
            "confirm.\n" + change.describe()
        )

    def _items(self, *args: str) -> tuple[list[dict] | None, str | None]:
        """`kubectl get ... -o json` items outside the protected namespaces."""
        ok, out = self.kubectl(*args, "-o", "json")
        if not ok:
            return None, f"Error: {out}"
        try:
            items = json.loads(out).get("items", [])
        except (json.JSONDecodeError, AttributeError) as e:
            return None, f"Error: unexpected kubectl output: {e}"
        return [i for i in items if (i.get("metadata") or {}).get("namespace") not in self.protected_namespaces], None

    def call(self, name: str, arguments: dict) -> str:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return f"Unknown tool '{name}'. Available: {', '.join(t['function']['name'] for t in TOOLS)}."
        try:
            return handler(**arguments)
        except TypeError as e:
            return f"Bad arguments for {name}: {e}"

    @staticmethod
    def _key(kind: str, name: str, namespace: str | None) -> tuple[str, str, str]:
        return kind.lower(), name, namespace or "default"

    def tool_list_k8s_resource_names(self, resource_type: str = WORKLOADS, namespace: str | None = None) -> str:
        if (refused := self._protected(namespace) or self._invalid(resource_type=resource_type, namespace=namespace)):
            return refused
        where = f" in namespace '{namespace}'." if namespace else "."
        if resource_type.lower() == WORKLOADS:
            items, error = self.list_resources()
            if error:
                return f"Error: {error}"
            items = [i for i in items if i["namespace"] not in self.protected_namespaces and (not namespace or i["namespace"] == namespace)]
            return "\n".join(f"{i['kind']} {i['namespace']}/{i['name']}" for i in items) if items else "No workloads found" + where
        if canonical_kind(resource_type) == "secret":
            return SECRETS_REFUSED
        items, error = self._items("get", resource_type, *(["-n", namespace] if namespace else ["-A"]))
        if error:
            return error
        return "\n".join(f"{i.get('kind', resource_type)} {_where(i)}" for i in items) if items else f"No {resource_type} found" + where

    def tool_extract_k8s_resource_yamls(self, resource_type: str, namespace: str | None = None, label_selector: str | None = None) -> str:
        import yaml

        from demo.core import strip_runtime_noise

        namespace = namespace or "default"
        if (refused := self._protected(namespace) or self._invalid(resource_type=resource_type, namespace=namespace)):
            return refused
        if label_selector is not None and not _SELECTOR_RE.match(str(label_selector)):
            return f"Invalid label_selector {label_selector!r}: expected e.g. app=web."
        if canonical_kind(resource_type) == "secret":
            return SECRETS_REFUSED
        items, error = self._items("get", resource_type, "-n", namespace, *(["-l", label_selector] if label_selector else []))
        if error:
            return error
        if not items:
            return f"No {resource_type} found in namespace '{namespace}'" + (f" with labels {label_selector}." if label_selector else ".")
        shown = items[:MAX_EXTRACTED_OBJECTS]
        header = f"{len(items)} {resource_type} in {namespace}" + (f" (the first {len(shown)} below)" if len(items) > len(shown) else "") + ":\n"
        return header + redact_credentials(yaml.safe_dump_all([strip_runtime_noise(i) for i in shown], sort_keys=False))

    def tool_get_specific_object_yaml(self, kind: str, name: str, namespace: str | None = None) -> str:
        kind, name, namespace = self._key(kind, name, namespace)
        if (refused := self._protected(namespace) or self._invalid(kind=kind, name=name, namespace=namespace)):
            return refused
        if canonical_kind(kind) == "secret":
            return SECRETS_REFUSED
        text, error = self.fetch_yaml(kind, namespace, name)
        return f"Error: {error}" if error else redact_credentials(text)

    def tool_audit_manifest(self, kind: str, name: str, namespace: str | None = None) -> str:
        text, problem = self.audit_input(kind, name, namespace)
        if problem:
            return problem
        parsed, errors = self.audit(text)
        return self.record_audit(kind, name, namespace, text, parsed, errors)

    # audit_manifest in two halves, so the MCP tool layer can run the model
    # in the host between them (demo/mcp_client.py).
    def audit_input(self, kind: str, name: str, namespace: str | None = None) -> tuple[str | None, str | None]:
        """(the manifest to audit, None), or (None, why it can't be)."""
        key = self._key(kind, name, namespace)
        if (refused := self._protected(key[2]) or self._invalid(kind=key[0], name=key[1], namespace=key[2])):
            return None, refused
        text, error = self.fetch_yaml(key[0], key[2], key[1])
        return (None, f"Error: {error}") if error else (text, None)

    def record_audit(self, kind: str, name: str, namespace: str | None, text: str, parsed: dict | None, errors: list[str]) -> str:
        """Keeps the audit (apply_fix uses it) and describes it for the planner."""
        key = self._key(kind, name, namespace)
        if parsed is None:
            return "The audit failed (the auditing model's answer was invalid): " + "; ".join(errors)
        self.audits[key] = {"yaml": text, "result": parsed}
        findings = parsed["findings"]
        if not findings:
            return f"No problems found in {kind} {key[2]}/{name}."
        lines = [f"{len(findings)} problem(s) in {kind} {key[2]}/{name}:"]
        for n, f in enumerate(findings, 1):
            lines.append(f"{n}. {f['rule_id']} ({f['severity']}): {f['message']} [at {f['path']}]")
        if parsed["patch"]:
            extra = f", plus {len(parsed['new_resources'])} new resource(s) to create" if parsed["new_resources"] else ""
            lines.append(f"A fix with {len(parsed['patch'])} change(s){extra} is ready: call apply_fix to apply it.")
        return "\n".join(lines)

    def tool_apply_fix(self, kind: str, name: str, namespace: str | None = None) -> str:
        key = self._key(kind, name, namespace)
        if (refused := self._invalid(kind=key[0], name=key[1], namespace=key[2]) or self._change_refused(key[2])):
            return refused
        audit = self.audits.get(key)
        if audit is None:
            return f"No audit of {kind} {key[2]}/{name} yet: call audit_manifest first."
        result = audit["result"]
        if not result["patch"]:
            return "The latest audit has no fix to apply."
        patched, error = self.apply_patch(audit["yaml"], result["patch"])
        if error:
            return f"The fix could not be applied to the manifest: {error}"
        ok, out = self.kubectl("apply", "--dry-run=server", "-n", key[2], "-f", "-", input_text=patched)
        if not ok:
            return f"The fix failed server-side validation (dry run), nothing was changed: {out}"
        return self._propose(PendingChange(
            kind=kind, name=name, namespace=key[2], findings=result["findings"], original_yaml=audit["yaml"],
            patched_yaml=patched, new_resources=list(result["new_resources"]), dry_run_output=out.strip(),
            patch=list(result["patch"]), running_images=self.running_images(audit["yaml"], key[2]),
            request=self.last_request,
        ))

    def _parse_objects(self, manifest: str, namespace: str | None) -> tuple[list[dict] | None, str | None]:
        """The YAML's objects, each with metadata.namespace set; or an error."""
        import yaml

        try:
            docs = [d for d in yaml.safe_load_all(str(manifest)) if d is not None]
        except yaml.YAMLError as e:
            return None, f"Invalid YAML: {e}"
        if not docs:
            return None, "The YAML has no objects."
        for doc in docs:
            if not isinstance(doc, dict) or not doc.get("apiVersion") or not doc.get("kind") or not (doc.get("metadata") or {}).get("name"):
                return None, "Every object needs apiVersion, kind and metadata.name."
            if canonical_kind(str(doc["kind"])) in CLUSTER_SCOPED_KINDS:
                return None, f"Refused: {doc['kind']} is cluster-scoped; the agent only works with namespaced objects."
            meta = doc["metadata"]
            meta["namespace"] = meta.get("namespace") or namespace or "default"
            if (refused := self._protected(meta["namespace"]) or self._invalid(name=meta["name"], namespace=meta["namespace"])):
                return None, refused
        return docs, None

    def tool_validate_k8s_yaml_dry_run(self, manifest: str, namespace: str | None = None) -> str:
        import yaml

        docs, error = self._parse_objects(manifest, namespace)
        if error:
            return error if error.startswith("Refused") else f"Not valid: {error}"
        ok, out = self.kubectl("apply", "--dry-run=server", "-f", "-", input_text=yaml.safe_dump_all(docs, sort_keys=False))
        if not ok:
            return f"Not valid: the API server rejected it (server-side dry run, nothing was changed): {out}"
        # kubectl says "created (server dry run)" -- the planner took that as
        # done; say what WOULD happen instead.
        out = re.sub(r" (created|configured|unchanged) \(server dry run\)",
                     lambda m: {"created": " would be created", "configured": " would be updated", "unchanged": " is already up to date"}[m.group(1)],
                     out.strip())
        return f"Valid: the API server accepts it. This was only a check: nothing was created or changed.\n{out}"

    def tool_apply_yaml_to_cluster(self, manifest: str, namespace: str | None = None) -> str:
        import yaml

        docs, error = self._parse_objects(manifest, namespace)
        if error:
            return error
        if len(docs) != 1:
            return f"Send one object per call (this YAML has {len(docs)}): each change is confirmed and recorded on its own."
        doc = docs[0]
        kind, name, ns = canonical_kind(str(doc["kind"])), doc["metadata"]["name"], doc["metadata"]["namespace"]
        if kind == "secret":
            return SECRETS_REFUSED
        if _has_masked_value(doc):
            return ("Refused: the YAML contains masked values (like 'abcd***'), copied from a read tool's output. Applying it "
                    "would overwrite the real values; to fix a plaintext credential, use audit_manifest and apply_fix.")
        if (refused := self._change_refused(ns)):
            return refused
        current, error = self.fetch_yaml(kind, ns, name)
        if error and "not found" not in error.lower():
            return f"Error: {error}"
        if current is None:  # a new object: labelled, so a demo reset removes it
            doc = yaml.safe_load(_with_label(yaml.safe_dump(doc, sort_keys=False), self.created_label))
        else:  # both in sorted-key form, so the diff shows changes, not key order
            current = yaml.safe_dump(yaml.safe_load(current), sort_keys=True)
        new_yaml = yaml.safe_dump(doc, sort_keys=True)
        ok, out = self.kubectl("apply", "--dry-run=server", "-f", "-", input_text=new_yaml)
        if not ok:
            return f"The YAML failed server-side validation (dry run), nothing was changed: {out}"
        changed = sum(1 for line in unified_diff(current or "", new_yaml).splitlines() if line[:1] in "+-" and line[:3] not in ("+++", "---"))
        return self._propose(PendingChange(
            kind=kind, name=name, namespace=ns, findings=[], original_yaml=current or "", patched_yaml=new_yaml, new_resources=[],
            dry_run_output=out.strip(), request=self.last_request, action="apply",
            summary=f"{'create' if current is None else 'update'} {doc['kind']} {ns}/{name} from the given YAML ({changed} line(s) differ)",
        ))

    def tool_scale_resource(self, kind: str, name: str, namespace: str | None = None, replicas=None) -> str:
        import yaml

        key = self._key(kind, name, namespace)
        if (refused := self._invalid(kind=key[0], name=key[1], namespace=key[2])):
            return refused
        target = canonical_kind(kind)
        if target not in SCALABLE_KINDS:
            return f"A {kind} can't be scaled: only {', '.join(sorted(SCALABLE_KINDS))} have replicas."
        try:
            replicas = int(replicas)
        except (TypeError, ValueError):
            return f"replicas must be a whole number, not {replicas!r}."
        if not 0 <= replicas <= MAX_REPLICAS:
            return f"Refused: replicas must be between 0 and {MAX_REPLICAS} in this demo."
        if (refused := self._change_refused(key[2])):
            return refused
        text, error = self.fetch_yaml(target, key[2], name)
        if error:
            return f"Error: {error}"
        spec = (yaml.safe_load(text) or {}).get("spec") or {}
        current = spec.get("replicas", 1)
        if current == replicas:
            return f"{kind} {key[2]}/{name} already has {replicas} replica(s); nothing to change."
        patch = [{"doc": 0, "op": "replace" if "replicas" in spec else "add", "path": "/spec/replicas", "value": replicas}]
        patched, error = self.apply_patch(text, patch)
        if error:
            return f"Error: {error}"
        ok, out = self.kubectl("apply", "--dry-run=server", "-n", key[2], "-f", "-", input_text=patched)
        if not ok:
            return f"Scaling failed server-side validation (dry run), nothing was changed: {out}"
        return self._propose(PendingChange(
            kind=target, name=name, namespace=key[2], findings=[], original_yaml=text, patched_yaml=patched, new_resources=[],
            dry_run_output=out.strip(), patch=patch, request=self.last_request, action="scale",
            summary=f"scale {target} {key[2]}/{name} from {current} to {replicas} replica(s)",
        ))

    def _find_pods(self, name: str, namespace: str | None) -> tuple[list[dict], str]:
        """The pod called `name`; if there's none, the pods of the workload
        called `name` (their names start with "<name>-"). Without a
        namespace, every namespace the agent may see is searched -- the
        planner often leaves it out. ([], error) when nothing matches."""
        if namespace:
            ok, out = self.kubectl("get", "pod", name, "-n", namespace, "-o", "json")
            if ok:
                try:
                    return [json.loads(out)], ""
                except json.JSONDecodeError as e:
                    return [], f"Error: unexpected kubectl output: {e}"
            if "not found" not in out.lower():
                return [], f"Error: {out}"
        items, error = self._items("get", "pods", *(["-n", namespace] if namespace else ["-A"]))
        if error:
            return [], error
        names = lambda p: str((p.get("metadata") or {}).get("name", ""))  # noqa: E731
        exact = [p for p in items if names(p) == name]
        if exact:
            return exact, ""
        pods = [p for p in items if names(p).startswith(name + "-")]
        if not pods:
            return [], f"No pod or workload called '{name}' in " + (f"namespace '{namespace}'." if namespace else "any namespace the agent can see.")
        return pods, f"'{name}' is a workload; its pod(s): {', '.join(_where(p) for p in pods)}."

    def tool_get_pod_logs(self, pod_name: str, namespace: str | None = None, container: str | None = None,
                          tail_lines: int = 100, previous: bool = False) -> str:
        if (refused := self._protected(namespace) or self._invalid(pod_name=pod_name, namespace=namespace, container=container)):
            return refused
        try:
            tail = max(1, min(int(tail_lines), MAX_LOG_LINES))
        except (TypeError, ValueError):
            tail = 100
        pods, note = self._find_pods(pod_name, namespace)
        if not pods:
            return note
        pod, namespace = pods[0]["metadata"]["name"], pods[0]["metadata"].get("namespace") or namespace
        if len(pods) > 1:
            note += f" Showing {pod}."
        containers = [c["name"] for c in (pods[0].get("spec") or {}).get("containers") or []]
        which = ["-c", container] if container else (["--all-containers=true", "--prefix"] if len(containers) > 1 else [])
        args = ["logs", pod, "-n", namespace, f"--tail={tail}", *which] + (["--previous"] if _as_bool(previous) else [])
        ok, out = self.kubectl(*args)
        if not ok:
            return f"Error: {out}"
        header = f"{'Previous run of ' if _as_bool(previous) else ''}{namespace}/{pod}, last {tail} line(s):"
        return "\n".join(x for x in (note, header, mask_log_secrets(out.rstrip()) or "(no log output)") if x)

    def tool_get_pod_diagnostics(self, pod_name: str, namespace: str | None = None) -> str:
        if (refused := self._protected(namespace) or self._invalid(pod_name=pod_name, namespace=namespace)):
            return refused
        pods, note = self._find_pods(pod_name, namespace)
        if not pods:
            return note
        reports = [self._diagnose(pod, pod["metadata"].get("namespace") or namespace) for pod in pods[:3]]
        if len(pods) > 3:
            reports.append(f"(and {len(pods) - 3} more pod(s))")
        return "\n\n".join(x for x in (note, *reports) if x)

    def _diagnose(self, pod: dict, namespace: str) -> str:
        meta, spec, status = pod.get("metadata") or {}, pod.get("spec") or {}, pod.get("status") or {}
        name = meta.get("name", "?")
        lines = [f"Pod {namespace}/{name}: {status.get('phase', '?')}, node {spec.get('nodeName') or '(not scheduled)'}"
                 + (f", started {status['startTime']}" if status.get("startTime") else "")]
        if status.get("reason") or status.get("message"):
            lines.append(f"  pod status: {status.get('reason', '')} {status.get('message', '')}".rstrip())
        for c in status.get("conditions") or []:
            if c.get("status") != "True":
                lines.append(f"  condition {c.get('type')} is {c.get('status')}: {c.get('reason', '')} {c.get('message', '')}".rstrip())
        hints, reasons, exit_codes = [], set(), set()
        statuses = [("init container", s) for s in status.get("initContainerStatuses") or []]
        statuses += [("container", s) for s in status.get("containerStatuses") or []]
        for label, cs in statuses:
            state, last = cs.get("state") or {}, (cs.get("lastState") or {}).get("terminated")
            if "waiting" in state:
                now = f"waiting: {state['waiting'].get('reason', '')} {state['waiting'].get('message', '')}".rstrip()
                reasons.add(state["waiting"].get("reason"))
            elif "terminated" in state:
                t = state["terminated"]
                now = f"terminated: {t.get('reason', '')} (exit code {t.get('exitCode')})"
                reasons.add(t.get("reason"))
                exit_codes.add(t.get("exitCode"))
            else:
                now = f"running since {(state.get('running') or {}).get('startedAt', '?')}"
            lines.append(f"  {label} {cs.get('name')} ({cs.get('image')}): ready={cs.get('ready')}, restarts={cs.get('restartCount', 0)}, {now}")
            if last:
                lines.append(f"    last run: {last.get('reason', '')} (exit code {last.get('exitCode')}) at {last.get('finishedAt')}")
                reasons.add(last.get("reason"))
                exit_codes.add(last.get("exitCode"))
            if cs.get("restartCount", 0) and not any("previous=true" in h for h in hints):
                hints.append(f"it restarted: call get_pod_logs with previous=true for {name} to see why the last run ended")
        events, _ = self._items("get", "events", "-n", namespace, "--field-selector", f"involvedObject.name={name}")
        events = sorted(events or [], key=lambda e: str(e.get("lastTimestamp") or e.get("eventTime") or e.get("firstTimestamp") or ""))
        if events:
            lines.append(f"  recent events ({min(len(events), 8)} of {len(events)}):")
            for e in events[-8:]:
                lines.append(f"    {e.get('type', '')} {e.get('reason', '')} (x{e.get('count') or 1}): {e.get('message', '').strip()}")
                reasons.add(e.get("reason"))
        if "ContainerCannotRun" in reasons or exit_codes & {126, 127}:
            hints.insert(0, "the container's command can't start (exit code 127: the command or binary doesn't exist in the image; "
                            "126: it isn't executable): check command and args against the image -- there may be no logs")
        if reasons & {"ImagePullBackOff", "ErrImagePull", "InvalidImageName"}:
            hints.append("the image can't be pulled: check the image name and tag, and that the registry is reachable")
        if "OOMKilled" in reasons:
            hints.append("a container was killed for exceeding its memory limit: raise resources.limits.memory or reduce its usage")
        if "CreateContainerConfigError" in reasons:
            hints.append("a ConfigMap, Secret or key the pod references doesn't exist")
        if "FailedScheduling" in reasons:
            hints.append("the pod can't be scheduled: check its resource requests, node selectors and the nodes' free capacity")
        if reasons & {"Unhealthy"}:
            hints.append("a liveness or readiness probe is failing: check the probe's path, port and timing")
        if hints:
            lines.append("  hints:\n" + "\n".join(f"    - {h}" for h in hints))
        return "\n".join(lines)

    def running_images(self, manifest_yaml: str, namespace: str) -> dict:
        """{container name: "repo@sha256:..."} for the images the workload's
        pods run right now -- pinning to these keeps exactly the same image.
        Best effort: {} when the pods, their selector or a registry digest
        can't be found (e.g. a locally built image has no repo digest)."""
        import yaml

        doc = yaml.safe_load(manifest_yaml) or {}
        spec = doc.get("spec") or {}
        selector = (spec.get("selector") or {}).get("matchLabels") if doc.get("kind") != "Pod" else None
        if doc.get("kind") == "Pod":
            ok, out = self.kubectl("get", "pod", (doc.get("metadata") or {}).get("name", ""), "-n", namespace, "-o", "json")
            pods = [json.loads(out)] if ok else []
        elif isinstance(selector, dict) and selector:
            label = ",".join(f"{k}={v}" for k, v in selector.items())
            ok, out = self.kubectl("get", "pods", "-n", namespace, "-l", label, "-o", "json")
            pods = json.loads(out).get("items", []) if ok else []
        else:
            return {}
        images = {}
        for pod in pods:
            for status in (pod.get("status") or {}).get("containerStatuses") or []:
                image_id = str(status.get("imageID", "")).replace("docker-pullable://", "")
                if "@sha256:" in image_id and not image_id.startswith(("docker://", "sha256:")):
                    images.setdefault(status.get("name"), image_id)
        return images

    def preview(self, change: PendingChange, image_overrides: dict | None = None) -> tuple[str | None, str | None]:
        """The patched manifest with the user's image choices (patch index ->
        image) instead of the model's."""
        if not image_overrides:
            return change.patched_yaml, None
        patch = [dict(op) for op in change.patch]
        for index, image in image_overrides.items():
            patch[int(index)]["value"] = image.strip()
        return self.apply_patch(change.original_yaml, patch)

    def tool_run_kubectl(self, command: str) -> str:
        argv, reason = check_read_only(command)
        if argv is None:
            return f"Refused: {reason}"
        for namespace in _namespaces_in(argv):
            if (refused := self._protected(namespace)):
                return refused
        ok, out = self.kubectl(*argv)
        if not ok:
            return f"Error: {out}"
        if "-A" in argv or "--all-namespaces" in argv:
            # Hide protected namespaces' rows (the first column is NAMESPACE).
            out = "\n".join(line for line in out.splitlines() if line.split(" ", 1)[0] not in self.protected_namespaces)
        return out

    def _record(self, change: PendingChange, status: str, result: str, chosen: dict | None = None, created: list[str] | None = None) -> str:
        """Records the decided change in the audit log (if there is one);
        returns `result`, noting the record -- or that recording failed."""
        if self.audit_log is None:
            return result
        from demo.audit_log import Actor, build_change_record

        try:
            record = build_change_record(change, status, result, chosen, created)
            change_id = self.audit_log.record(self.actor or Actor("unknown", "unknown"), self.adapter, record)
        except Exception as e:  # noqa: BLE001 -- report it, never lose the outcome
            return f"{result} WARNING: this was NOT recorded in the audit log: {e}"
        return f"{result} (audit log: change #{change_id})"

    def decline_pending(self) -> str:
        change = self.pending
        self.pending = None
        if change is None:
            return "There was no pending change."
        return self._record(change, "declined", DECLINED)

    def execute_pending(self, new_resources: list[str] | None = None, image_overrides: dict | None = None) -> str:
        """Applies the confirmed change: new resources first (the fix may
        reference a Secret they create), then the patched manifest -- with
        the user's image choices, re-validated by a dry run, if they changed
        any. Every outcome is recorded in the audit log; while the log is
        unreachable nothing is applied (the change stays pending)."""
        change = self.pending
        self.pending = None
        if change is None:
            return "There was no pending change."
        if self.audit_log is not None and not self.audit_log.ping():
            self.pending = change
            return "Not applied: the audit log (PostgreSQL) is unreachable, and changes are only made when they can be recorded."
        chosen = {i: v for i, v in (image_overrides or {}).items() if v.strip() != change.patch[int(i)]["value"]}
        if chosen:
            if not all(is_pinned(v) for v in chosen.values()):
                return self._record(change, "rejected", "Not applied: every image must be pinned to a tag (not 'latest') or a digest.")
            patched, error = self.preview(change, chosen)
            if error:
                return self._record(change, "rejected", f"Not applied: the chosen image could not be set: {error}")
            ok, out = self.kubectl("apply", "--dry-run=server", "-n", change.namespace, "-f", "-", input_text=patched)
            if not ok:
                return self._record(change, "rejected", f"Not applied: the change with the chosen image failed the dry run: {out}")
            change.patched_yaml = patched
        chosen = {int(i): v.strip() for i, v in chosen.items()}
        if not self.can_apply:
            return self._record(change, "rejected", "Not applied: the logged-in user isn't an operator.", chosen)
        outputs, created = [], []
        for res in new_resources if new_resources is not None else change.new_resources:
            res = _with_label(res, self.created_label)
            ok, out = self.kubectl("apply", "-n", change.namespace, "-f", "-", input_text=res)
            if not ok:
                return self._record(change, "failed", f"Creating a new resource failed, the fix was NOT applied: {out}", chosen, created)
            outputs.append(out.strip())
            created.append(res)
        ok, out = self.kubectl("apply", "-n", change.namespace, "-f", "-", input_text=change.patched_yaml)
        if not ok:
            return self._record(change, "failed", f"Applying the {'fix' if change.action == 'fix' else 'change'} failed: {out}", chosen, created)
        outputs.append(out.strip())
        if change.action == "apply" and not change.original_yaml:
            created.append(change.patched_yaml)
        self.audits.pop(self._key(change.kind, change.name, change.namespace), None)
        images = [f"{op['container']} image set to {chosen.get(op['index'], op['value'])}" for op in change.image_ops]
        return self._record(change, "applied", "Applied: " + "; ".join(o for o in outputs + images if o), chosen, created)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@dataclass
class Event:
    type: str  # "tool_call" | "tool_result" | "confirm" | "final"
    name: str = ""
    arguments: dict = field(default_factory=dict)
    text: str = ""


# Small planners often announce an action instead of taking it ("Let me
# check...", "I will now call apply_fix") or stop after an audit the user
# asked to fix -- both seen on end-to-end runs with Qwen2.5-Coder-7B, and a
# second request in words didn't reliably help. Two mechanisms that do:
#  - a fix the user asked for, with an audit that has one ready, is called by
#    the loop itself (it still waits for the user's confirmation);
#  - an announced-but-not-made call is retried with the reply pre-filled with
#    "<tool_call>", so the model can only continue by writing the call.
_PROMISE_RE = re.compile(r"\b(let me|let's|i'll|i will|i'm going to|i am going to|please wait|first,? i)\b", re.IGNORECASE)
_FIX_REQUEST_RE = re.compile(r"\b(fix|repair|remediate|correct|corrig\w*|apply)\b", re.IGNORECASE)
# Change tools run only when the user's message asks for a change -- the
# 7B planner otherwise "helps" with unasked ones while diagnosing (a scale
# to the same count, an apply_fix without an audit), seen on a live run.
_CHANGE_REQUEST_RE = re.compile(
    r"\b(fix\w*|repair|remediate|correct|corrig\w*|apply|create|deploy|scale|update|set|change|add|remove|increase|decrease|"
    r"raise|reduce|patch|install|replicas?)\b",
    re.IGNORECASE,
)
NOT_ASKED = (
    "Not done: the user didn't ask for a change to the cluster. Answer their question from what the tools returned; "
    "you may suggest a change in words, and make it only if they ask."
)
_APPLY_REQUEST_RE = re.compile(r"\b(create|apply|deploy|add|install)\b", re.IGNORECASE)
_FENCED_YAML_RE = re.compile(r"```(?:ya?ml)?[ \t]*\n(.*?)```", re.DOTALL)
_NAMESPACE_IN_TEXT_RE = re.compile(
    r"\b(?:in|into|to|on)\s+(?:the\s+)?[`'\"]?([a-z0-9][-a-z0-9]*)[`'\"]?\s+namespace\b|\bnamespace\s+[`'\"]?([a-z0-9][-a-z0-9]*)",
    re.IGNORECASE,
)


def yaml_in_message(text: str) -> tuple[str | None, str]:
    """(the YAML object(s) in a chat message, the message without them):
    a ```yaml block, or everything from the first `apiVersion:`/`kind:`
    line. (None, text) when there's no parseable Kubernetes object."""
    import yaml

    match = _FENCED_YAML_RE.search(text)
    if match:
        candidate, rest = match.group(1), text.replace(match.group(0), "")
    else:
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines) if line.startswith(("apiVersion:", "kind:"))), None)
        if start is None:
            return None, text
        candidate, rest = "\n".join(lines[start:]), "\n".join(lines[:start])
    try:
        docs = [d for d in yaml.safe_load_all(candidate) if d is not None]
    except yaml.YAMLError:
        return None, text
    if not docs or not all(isinstance(d, dict) and d.get("kind") for d in docs):
        return None, text
    return candidate, rest


# ...or asks to check an object for problems, which only the auditor does.
_CHECK_REQUEST_RE = re.compile(
    r"\b(check|audit|scan|review|fix|repair|secur\w*|problems?|issues?|vulnerab\w*|misconfig\w*|verif\w*|corrig\w*|analis\w*)\b",
    re.IGNORECASE,
)
# ...or tells the user to run a kubectl command it could run itself.
_SUGGESTED_COMMAND_RE = re.compile(r"```[a-z]*\s*kubectl\s", re.IGNORECASE)
TOOL_CALL_PREFILL = "<tool_call>\n"
DECLINED = "The user declined the change; nothing was applied. Do not propose this change again unless the user asks for it."


class Agent:
    """planner(messages, tools, prefill="") -> the model's raw answer text
    (starting with `prefill`, which the model continues)."""

    def __init__(self, planner: Callable[..., str], toolbox: Toolbox, max_steps: int = MAX_STEPS):
        self.planner = planner
        self.toolbox = toolbox
        # The tool definitions the planner sees: the in-process TOOLS, or the
        # ones the MCP tool layer discovered from its server.
        self.tools = getattr(toolbox, "tools", None) or TOOLS
        self.max_steps = max_steps
        self.history: list[dict] = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
        self._turn_tools: list[str] = []
        self._turn_calls: list[dict] = []

    @property
    def waiting(self) -> bool:
        """A change is waiting for the user's Confirm/Decline. It stays
        pending while the user asks questions about it."""
        return self.toolbox.pending is not None

    def ask(self, text: str):
        """Runs the loop for a new user message; yields Events. A pending
        change stays pending: the user can ask about it before deciding."""
        self.history.append({"role": "user", "content": text})
        self.toolbox.last_request = text
        yield from self._loop()

    def resume(self, approved: bool, new_resources: list[str] | None = None, image_overrides: dict | None = None):
        """The user clicked Confirm or Decline: applies the pending change if
        confirmed, tells the planner the outcome, and lets it continue (e.g.
        re-audit to verify)."""
        change = self.toolbox.pending
        if change is None:
            return
        if approved:
            result = self.toolbox.execute_pending(new_resources, image_overrides)
        else:
            result = self.toolbox.decline_pending()
        decision = "confirmed" if approved else "declined"
        self.history.append({
            "role": "user",
            "content": f"[I {decision} the proposed change to {change.kind} {change.namespace}/{change.name}. Result: {result}]",
        })
        yield Event("tool_result", name="apply_fix", text=result)
        # Same turn: apply_fix already happened (confirmed or declined), so
        # the "a fix was requested" nudge must not propose it again.
        self._turn_tools.append("apply_fix")
        yield from self._loop(new_turn=False)

    def _user_request(self) -> str:
        """The user's latest message itself (not a Confirm/Decline note)."""
        for message in reversed(self.history):
            if message["role"] == "user" and not message["content"].startswith(("[I confirmed", "[I declined")):
                return message["content"]
        return ""

    def _requested_apply_call(self) -> dict | None:
        """apply_yaml_to_cluster for the YAML in the user's message, when they
        asked to create/apply it and the planner answered in text instead
        (it asked for confirmation itself, on a live run)."""
        request = self._user_request()
        if MUTATING_TOOLS & set(self._turn_tools) or self.toolbox.pending is not None:
            return None
        manifest, rest = yaml_in_message(request)
        if manifest is None or not _APPLY_REQUEST_RE.search(rest):
            return None
        arguments = {"manifest": manifest}
        if (match := _NAMESPACE_IN_TEXT_RE.search(rest)):
            arguments["namespace"] = (match.group(1) or match.group(2)).lower()
        return {"name": "apply_yaml_to_cluster", "arguments": arguments}

    def _requested_audit_call(self) -> dict | None:
        """audit_manifest for the object the planner just read, when the
        user asked to check or fix it and the planner reviewed the YAML
        itself instead of calling the auditor -- seen with the 7B planner
        once get_specific_object_yaml existed."""
        user_turns = [m["content"] for m in self.history if m["role"] == "user"]
        if not user_turns or not _CHECK_REQUEST_RE.search(user_turns[-1]) or "audit_manifest" in self._turn_tools:
            return None
        if self.toolbox.pending is not None:
            return None
        reads = [c for c in self._turn_calls if c["name"] == "get_specific_object_yaml"]
        if not reads:
            return None
        args = {k: v for k, v in reads[-1]["arguments"].items() if k in ("kind", "name", "namespace")}
        return {"name": "audit_manifest", "arguments": args}

    def _requested_fix_call(self) -> dict | None:
        """apply_fix for the latest audit with a fix ready, when the user's
        message asked for a fix and it hasn't been attempted this turn."""
        user_turns = [m["content"] for m in self.history if m["role"] == "user"]
        if not user_turns or not _FIX_REQUEST_RE.search(user_turns[-1]) or MUTATING_TOOLS & set(self._turn_tools):
            return None
        if self.toolbox.pending is not None:  # questions about a pending fix often mention "fix"
            return None
        ready = self.toolbox.ready_fixes()
        if not ready:
            return None
        kind, name, namespace = ready[-1]
        return {"name": "apply_fix", "arguments": {"kind": kind, "name": name, "namespace": namespace}}

    def _loop(self, new_turn: bool = True):
        if new_turn:
            self._turn_tools = []
            self._turn_calls = []
        for _ in range(self.max_steps):
            raw = self.planner(self.history, self.tools)
            calls, text, errors = parse_tool_calls(raw)
            if not calls and not errors:
                if (forced := self._requested_audit_call() or self._requested_fix_call() or self._requested_apply_call()) is not None:
                    calls = [forced]
                elif _PROMISE_RE.search(text) or _SUGGESTED_COMMAND_RE.search(text):
                    retry_calls, _, _ = parse_tool_calls(self.planner(self.history, self.tools, prefill=TOOL_CALL_PREFILL))
                    if retry_calls:
                        calls = retry_calls
            if not calls and not errors:
                self.history.append({"role": "assistant", "content": text})
                yield Event("final", text=text)
                return
            self.history.append(
                {"role": "assistant", "content": text, "tool_calls": [{"type": "function", "function": c} for c in calls]}
            )
            for error in errors:
                self.history.append({"role": "tool", "name": "error", "content": f"Invalid tool call: {error}"})
            for call in calls:
                self._turn_tools.append(call["name"])
                self._turn_calls.append(call)
                yield Event("tool_call", name=call["name"], arguments=call["arguments"])
                before = self.toolbox.pending
                if call["name"] in MUTATING_TOOLS and not _CHANGE_REQUEST_RE.search(self._user_request()):
                    result = NOT_ASKED
                else:
                    result = self.toolbox.call(call["name"], call["arguments"])
                self.history.append({"role": "tool", "name": call["name"], "content": _truncate(result)})
                if self.toolbox.pending is not None and self.toolbox.pending is not before:
                    # A new change waits for the user; the planner gets one
                    # more step to explain it, without calling more tools.
                    yield Event("confirm", name=call["name"], arguments=call["arguments"], text=result)
                    yield from self._explain_pending()
                    return
                yield Event("tool_result", name=call["name"], text=result)
        final = f"I stopped after {self.max_steps} steps without finishing. Try a more specific request."
        self.history.append({"role": "assistant", "content": final})
        yield Event("final", text=final)

    def _explain_pending(self):
        """The planner's short explanation of the change it just proposed."""
        _, text, _ = parse_tool_calls(self.planner(self.history, self.tools))
        final = text or "I've prepared a change; review it below and confirm or decline."
        self.history.append({"role": "assistant", "content": final})
        yield Event("final", text=final)


# ---------------------------------------------------------------------------
# Real model wiring
# ---------------------------------------------------------------------------


def make_planner(model, tokenizer, max_seq_length: int = 4096, max_new_tokens: int = 512):
    """The planner: the base instruct model (LoRA adapter disabled) with the
    tool definitions in its own chat template. Oldest turns are dropped when
    the conversation no longer fits the context window."""
    import torch

    def planner(messages: list[dict], tools: list[dict], prefill: str = "") -> str:
        kept = list(messages)
        while True:
            prompt = tokenizer.apply_chat_template(kept, tools=tools, tokenize=False, add_generation_prompt=True) + prefill
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            if ids.shape[1] + max_new_tokens <= max_seq_length or len(kept) <= 2:
                break
            # Drop the oldest exchange after the system prompt, keeping
            # tool results attached to the call that produced them.
            del kept[1]
            while len(kept) > 2 and kept[1]["role"] == "tool":
                del kept[1]
        ids = ids.to(model.device)
        with model.disable_adapter(), torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
        for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            text = text.split(marker)[0]
        return (prefill + text).strip()

    return planner


def make_auditor(model, tokenizer, max_new_tokens: int = 1280):
    """The auditor: the same model with the fine-tuned adapter enabled."""
    from dataset.schema import SYSTEM_PROMPT
    from finetune.evaluate import generate_response_text, parse_model_output

    def audit(manifest_yaml: str) -> tuple[dict | None, list[str]]:
        raw = generate_response_text(model, tokenizer, SYSTEM_PROMPT, manifest_yaml, max_new_tokens)
        return parse_model_output(raw)

    return audit


TOOL_LAYERS = ("function", "mcp")


def build_agent(model, tokenizer, max_seq_length: int = 4096, tool_layer: str = "function") -> Agent:
    """`tool_layer` "function": the tools run in this process (Toolbox);
    "mcp": they run in demo/mcp_server.py, reached over MCP (McpToolbox)."""
    from demo import core

    if tool_layer not in TOOL_LAYERS:
        raise ValueError(f"unknown tool layer {tool_layer!r}: use one of {', '.join(TOOL_LAYERS)}")
    if tool_layer == "mcp":
        from demo.mcp_client import McpToolbox

        return Agent(make_planner(model, tokenizer, max_seq_length), McpToolbox(audit=make_auditor(model, tokenizer)))
    toolbox = Toolbox(
        audit=make_auditor(model, tokenizer),
        kubectl=core.kubectl,
        list_resources=core.list_resources,
        fetch_yaml=core.fetch_resource_yaml,
        apply_patch=core.apply_patch_to_manifest,
        protected_namespaces=frozenset(core.PROTECTED_NAMESPACES),
        created_label=core.AGENT_LABEL,
    )
    return Agent(make_planner(model, tokenizer, max_seq_length), toolbox)
