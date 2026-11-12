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

Guardrails:
  - read-only tools run freely; `run_kubectl` only accepts read-only verbs
    and never prints Secret contents;
  - the only way to change the cluster is `apply_fix`, which validates the
    change with a server-side dry run and then PAUSES the agent until the
    user confirms or declines it in the UI (Agent.resume).

The planner and the tools are injected (Agent(planner, toolbox)), so the
loop is tested with a scripted fake planner and a fake kubectl -- see
demo/test_agent.py.
"""

from __future__ import annotations

import difflib
import json
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

MAX_STEPS = 8
MAX_TOOL_RESULT_CHARS = 2500

PLANNER_SYSTEM_PROMPT = """\
You are a Kubernetes cluster agent. You help the user inspect their cluster \
and find and fix security and configuration problems in its workloads.

How to work:
- Use the tools to get facts. Never invent cluster state or audit results: \
only report what the tools return.
- To check a workload for problems, call audit_manifest. It runs a \
specialized auditing model and returns the problems it found and whether a \
fix is ready.
- To fix the problems an audit found, call apply_fix. The change is \
validated with a server-side dry run and then shown to the user, who \
confirms or declines it -- do not ask for confirmation yourself. If the \
user asked you to fix something, call apply_fix right after the audit \
reports a fix is ready.
- After a fix is applied, call audit_manifest again to verify it.
- For other questions about the cluster, use list_resources or run_kubectl \
(read-only commands only).
- When you need a tool, call it right away, in this same answer. Never reply with "please wait" or describe a call without making it. A call looks like this:
<tool_call>
{"name": "audit_manifest", "arguments": {"kind": "Deployment", "name": "web", "namespace": "demo"}}
</tool_call>
- If the user is just chatting, answer without tools.
- Keep answers short and factual."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_resources",
            "description": "List the workloads (pods, deployments, statefulsets, daemonsets, replicasets, jobs, cronjobs) in the cluster.",
            "parameters": {
                "type": "object",
                "properties": {"namespace": {"type": "string", "description": "Only this namespace. Omit for all namespaces."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_manifest",
            "description": "Get the YAML manifest of one resource.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "description": "e.g. Deployment"},
                    "name": {"type": "string"},
                    "namespace": {"type": "string", "description": "Defaults to 'default'."},
                },
                "required": ["kind", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_manifest",
            "description": "Check one resource for security and configuration problems (plaintext credentials, unpinned images, "
            "privileged containers, host access, broken selectors, probe and resource mistakes, misspelled names...). "
            "Returns the problems found and whether a fix is ready.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "description": "e.g. Deployment"},
                    "name": {"type": "string"},
                    "namespace": {"type": "string", "description": "Defaults to 'default'."},
                },
                "required": ["kind", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_fix",
            "description": "Apply the fix from the latest audit of a resource. It is validated with a server-side dry run, "
            "then the user is asked to confirm before anything changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "name": {"type": "string"},
                    "namespace": {"type": "string", "description": "Defaults to 'default'."},
                },
                "required": ["kind", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_kubectl",
            "description": "Run a READ-ONLY kubectl command (get, describe, logs, top, events, explain, version, ...). "
            "Commands that change the cluster are refused.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "e.g. kubectl get pods -n demo"}},
                "required": ["command"],
            },
        },
    },
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

    @property
    def diff(self) -> str:
        return unified_diff(self.original_yaml, self.patched_yaml)

    def describe(self) -> str:
        """The change in plain terms, for the planner -- so it can answer the
        user's questions about it. Never repeats a plaintext value that's
        being removed (it may be a credential)."""
        import yaml

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

    def tool_list_resources(self, namespace: str | None = None) -> str:
        items, error = self.list_resources()
        if error:
            return f"Error: {error}"
        if (refused := self._protected(namespace)):
            return refused
        items = [i for i in items if i["namespace"] not in self.protected_namespaces]
        if namespace:
            items = [i for i in items if i["namespace"] == namespace]
        if not items:
            return "No workloads found" + (f" in namespace '{namespace}'." if namespace else ".")
        return "\n".join(f"{i['kind']} {i['namespace']}/{i['name']}" for i in items)

    def tool_get_manifest(self, kind: str, name: str, namespace: str | None = None) -> str:
        kind, name, namespace = self._key(kind, name, namespace)
        if (refused := self._protected(namespace)):
            return refused
        text, error = self.fetch_yaml(kind, namespace, name)
        return f"Error: {error}" if error else text

    def tool_audit_manifest(self, kind: str, name: str, namespace: str | None = None) -> str:
        key = self._key(kind, name, namespace)
        if (refused := self._protected(key[2])):
            return refused
        text, error = self.fetch_yaml(key[0], key[2], key[1])
        if error:
            return f"Error: {error}"
        parsed, errors = self.audit(text)
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
        if (refused := self._protected(key[2])):
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
                "Tell the user the fix is available but needs an operator."
            )
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
        self.pending = PendingChange(
            kind=kind, name=name, namespace=key[2], findings=result["findings"], original_yaml=audit["yaml"],
            patched_yaml=patched, new_resources=list(result["new_resources"]), dry_run_output=out.strip(),
            patch=list(result["patch"]), running_images=self.running_images(audit["yaml"], key[2]),
            request=self.last_request,
        )
        return (
            "The fix passed the server-side dry run and is waiting for the user's confirmation (the user clicks Confirm or "
            "Decline). Briefly tell the user what will change. Answer their questions about it; nothing is applied until they "
            "confirm.\n" + self.pending.describe()
        )

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
            return self._record(change, "failed", f"Applying the fix failed: {out}", chosen, created)
        outputs.append(out.strip())
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
# ...or tells the user to run a kubectl command it could run itself.
_SUGGESTED_COMMAND_RE = re.compile(r"```[a-z]*\s*kubectl\s", re.IGNORECASE)
TOOL_CALL_PREFILL = "<tool_call>\n"
DECLINED = "The user declined the change; nothing was applied. Do not propose this fix again unless the user asks for it."


class Agent:
    """planner(messages, tools, prefill="") -> the model's raw answer text
    (starting with `prefill`, which the model continues)."""

    def __init__(self, planner: Callable[..., str], toolbox: Toolbox, max_steps: int = MAX_STEPS):
        self.planner = planner
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.history: list[dict] = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
        self._turn_tools: list[str] = []

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

    def _requested_fix_call(self) -> dict | None:
        """apply_fix for the latest audit with a fix ready, when the user's
        message asked for a fix and it hasn't been attempted this turn."""
        user_turns = [m["content"] for m in self.history if m["role"] == "user"]
        if not user_turns or not _FIX_REQUEST_RE.search(user_turns[-1]) or "apply_fix" in self._turn_tools:
            return None
        if self.toolbox.pending is not None:  # questions about a pending fix often mention "fix"
            return None
        ready = [key for key, a in self.toolbox.audits.items() if a["result"]["patch"]]
        if not ready:
            return None
        kind, name, namespace = ready[-1]
        return {"name": "apply_fix", "arguments": {"kind": kind, "name": name, "namespace": namespace}}

    def _loop(self, new_turn: bool = True):
        if new_turn:
            self._turn_tools = []
        for _ in range(self.max_steps):
            raw = self.planner(self.history, TOOLS)
            calls, text, errors = parse_tool_calls(raw)
            if not calls and not errors:
                if (forced := self._requested_fix_call()) is not None:
                    calls = [forced]
                elif _PROMISE_RE.search(text) or _SUGGESTED_COMMAND_RE.search(text):
                    retry_calls, _, _ = parse_tool_calls(self.planner(self.history, TOOLS, prefill=TOOL_CALL_PREFILL))
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
                yield Event("tool_call", name=call["name"], arguments=call["arguments"])
                before = self.toolbox.pending
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
        _, text, _ = parse_tool_calls(self.planner(self.history, TOOLS))
        final = text or "I've prepared a fix; review the changes below and confirm or decline."
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


def build_agent(model, tokenizer, max_seq_length: int = 4096) -> Agent:
    from demo import core

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
