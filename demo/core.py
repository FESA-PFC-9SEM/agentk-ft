"""
Shared logic for the demo UIs (streamlit_app.py and app.py) and the agent
(agent.py): model loading, kubectl integration, and patch application, kept
in one place so fetching, patching and applying behave the same everywhere.
The manifest-security model's helpers (load_model / generate_response_text /
parse_model_output) are re-exported from finetune/evaluate.py rather than
duplicated here.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import jsonpatch
import yaml

from dataset.schema import RULES, SYSTEM_PROMPT  # noqa: F401  (re-exported for callers)
from finetune.evaluate import generate_response_text, load_model, parse_model_output, patch_ops_for_finding, patch_ops_for_findings  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO_ROOT / "scenarios"
RUNS_DIR = REPO_ROOT / "runs"
NON_MANIFEST_SCENARIO_FILES = {"test_cases.yaml", "ground_truth.yaml"}

# Kinds dataset/k8s.py's get_pod_spec() actually knows how to find a PodSpec
# in -- the same scope the model was trained on. ReplicationController and
# Service are deliberately excluded: the model was never trained to find (or
# not find) issues in them, so surfacing one would look like a false negative
# instead of the honest scope boundary it is.
RESOURCE_KINDS = ["pod", "deployment", "statefulset", "daemonset", "replicaset", "job", "cronjob"]
SYSTEM_NAMESPACES = {"kube-system", "kube-node-lease", "kube-public"}
# The namespaces the agent works on (its service account can change them;
# deploy/agent-rbac.yaml) and the ones it must never touch: the login
# system, its own identity and the audit database that records what it
# changes. The agent's tools also refuse and hide the protected ones
# (demo/agent.py).
MANAGED_NAMESPACES = [ns for ns in os.environ.get("AGENT_NAMESPACES", "default,demo,prod").split(",") if ns]
PROTECTED_NAMESPACES = {"keycloak", "agent-system", "audit"}
# Every resource the agent creates carries this label, so a reset can remove
# exactly those (demo/reset_cluster.py).
AGENT_LABEL = ("app.kubernetes.io/managed-by", "cluster-agent")

_DROP_METADATA_KEYS = (
    "managedFields", "resourceVersion", "uid", "creationTimestamp",
    "generation", "selfLink", "annotations",
)


def kubectl(*args: str, timeout: float = 10.0, input_text: str | None = None) -> tuple[bool, str]:
    """Runs kubectl as the agent: with the agent's own service-account
    kubeconfig when AGENT_KUBECONFIG is set (demo/run_secure.sh sets it), so
    the cluster's RBAC -- not just this code -- limits what it can change."""
    kubeconfig = os.environ.get("AGENT_KUBECONFIG")
    prefix = ["kubectl", "--kubeconfig", kubeconfig] if kubeconfig else ["kubectl"]
    try:
        result = subprocess.run(
            [*prefix, *args], capture_output=True, text=True, timeout=timeout, input=input_text
        )
    except FileNotFoundError:
        return False, "kubectl not found on PATH"
    except subprocess.TimeoutExpired:
        return False, f"kubectl {' '.join(args)} timed out"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, result.stdout


def kubectl_context() -> str | None:
    ok, out = kubectl("config", "current-context", timeout=5)
    return out.strip() if ok else None


def cluster_reachable() -> bool:
    ok, _ = kubectl("cluster-info", timeout=5)
    return ok


def strip_runtime_noise(doc: dict) -> dict:
    """A live `kubectl get -o yaml` carries a `status` block and
    cluster-assigned metadata the training data never had -- strip it so the
    fetched manifest looks like the spec-only YAML the model was trained on."""
    doc = dict(doc)
    meta = dict(doc.get("metadata") or {})
    for key in _DROP_METADATA_KEYS:
        meta.pop(key, None)
    doc["metadata"] = meta
    doc.pop("status", None)
    return doc


def list_resources(include_system: bool = False) -> tuple[list[dict], str | None]:
    """Returns (items, error). items is [] on error."""
    ok, out = kubectl("get", ",".join(RESOURCE_KINDS), "-A", "-o", "json", timeout=6)
    if not ok:
        return [], out
    items = []
    try:
        for item in json.loads(out).get("items", []):
            namespace = item["metadata"]["namespace"]
            if namespace in PROTECTED_NAMESPACES or (not include_system and namespace in SYSTEM_NAMESPACES):
                continue
            items.append({"kind": item["kind"], "namespace": namespace, "name": item["metadata"]["name"]})
    except (json.JSONDecodeError, KeyError) as e:
        return [], f"unexpected kubectl output: {e}"
    items.sort(key=lambda x: (x["namespace"], x["kind"], x["name"]))
    return items, None


def fetch_resource_yaml(kind: str, namespace: str, name: str) -> tuple[str | None, str | None]:
    """Returns (yaml_text, error)."""
    ok, out = kubectl("get", kind, name, "-n", namespace, "-o", "yaml")
    if not ok:
        return None, out
    try:
        text = yaml.safe_dump(
            strip_runtime_noise(yaml.safe_load(out)), sort_keys=False, default_flow_style=False, allow_unicode=True
        )
    except yaml.YAMLError:
        text = out  # fall back to the raw kubectl output rather than fail
    return text, None


def apply_yaml_to_cluster(yaml_text: str, namespace: str | None = None) -> tuple[bool, str]:
    """`kubectl apply -f -` the given YAML (piped via stdin) -- pushes a fix
    (or a brand-new resource such as an externalized Secret) back to the live
    cluster. `namespace`, if given, is passed as -n (useful when the YAML
    itself doesn't set metadata.namespace, e.g. a model-generated new_resource)."""
    args = ["apply", "-f", "-"]
    if namespace:
        args += ["-n", namespace]
    return kubectl(*args, timeout=20, input_text=yaml_text)


def list_scenarios() -> list[str]:
    return sorted(p.name for p in SCENARIOS_DIR.glob("*.yaml") if p.name not in NON_MANIFEST_SCENARIO_FILES)


def load_scenario(name: str) -> tuple[str | None, list[dict]]:
    """Returns (yaml_text_or_None, ground_truth). ground_truth is read from
    scenarios/test_cases.yaml's hand-labeled entries for this file, if any."""
    if "/" in name or "\\" in name or name in NON_MANIFEST_SCENARIO_FILES:
        return None, []
    path = SCENARIOS_DIR / name
    if not path.is_file():
        return None, []

    ground_truth = []
    test_cases_path = SCENARIOS_DIR / "test_cases.yaml"
    if test_cases_path.is_file():
        all_cases = yaml.safe_load(test_cases_path.read_text(encoding="utf-8")) or {}
        for case in all_cases.get(name, []) or []:
            ground_truth.append(
                {"category": case.get("category"), "description": case.get("description"), "rule_id": case.get("rule_id")}
            )
    return path.read_text(encoding="utf-8"), ground_truth


# The only two `strategy` values finetune/train_unsloth.py (this task's
# trainer) writes to run_info.json. `runs/` is shared with other training
# tasks in this repo (e.g. monitoring_training's incident-diagnosis model,
# tagged "task": "monitoring_training / ...") and a few pre-run_info.json
# ad-hoc experiments -- filtering on this is what keeps those out of this
# demo's adapter picker, since they use a different system prompt/output
# schema entirely and would just produce garbage here.
_THIS_TASK_STRATEGIES = {"single-defect", "multi-defect"}


def _run_info(adapter_path: Path) -> dict | None:
    try:
        return json.loads((adapter_path.parent / "run_info.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_adapters() -> list[str]:
    """Every runs/*/lora_adapter belonging to THIS task (see
    _THIS_TASK_STRATEGIES), repo-relative, newest first by directory mtime
    (run folder names aren't consistently timestamp-prefixed across the
    whole shared runs/ directory, so mtime is the only reliable order)."""
    if not RUNS_DIR.is_dir():
        return []
    candidates = (p for p in RUNS_DIR.glob("*/lora_adapter") if p.is_dir())
    matching = [p for p in candidates if (info := _run_info(p)) and info.get("strategy") in _THIS_TASK_STRATEGIES]
    matching.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    adapters = [str(p.relative_to(REPO_ROOT)) for p in matching]
    # The demo's model is chosen explicitly (DEMO_ADAPTER, set by
    # demo/run_secure.sh), not "newest first": copying in a new or broken
    # run must never silently replace it.
    preferred = os.environ.get("DEMO_ADAPTER", "").rstrip("/")
    if preferred in adapters:
        adapters.remove(preferred)
        adapters.insert(0, preferred)
    return adapters


def base_model_for_adapter(adapter_path: str) -> str | None:
    info = _run_info(Path(adapter_path))
    return info.get("model") if info else None


def classify_intent(text: str) -> str:
    """"manifest" if `text` parses as YAML with a top-level `kind` field (a
    pasted Kubernetes object, audited directly), else "cluster_action" (a
    natural-language request for the agent)."""
    try:
        first_doc = next(yaml.safe_load_all(text), None)
    except yaml.YAMLError:
        first_doc = None
    if isinstance(first_doc, dict) and "kind" in first_doc:
        return "manifest"
    return "cluster_action"


def _patch_ops_without_doc_key(patch_ops: list[dict]) -> list[dict]:
    return [{k: v for k, v in op.items() if k != "doc"} for op in patch_ops]


def apply_patch_to_manifest(manifest_text: str, patch_ops: list[dict]) -> tuple[str | None, str | None]:
    """Applies a subset (or all) of dataset/schema.py's {"doc","op","path",
    "value"} patch ops to the (possibly multi-document) input manifest.
    Returns (patched_yaml, None) on success or (None, error) on failure --
    the same round-trip dataset/build.py checks at generation time."""
    try:
        docs = list(yaml.safe_load_all(manifest_text))
    except yaml.YAMLError as e:
        return None, f"could not parse the original manifest: {e}"

    by_doc: dict[int, list[dict]] = {}
    for op in patch_ops:
        by_doc.setdefault(op.get("doc", 0), []).append(op)

    try:
        for idx, ops in by_doc.items():
            if 0 <= idx < len(docs):
                docs[idx] = jsonpatch.apply_patch(docs[idx], _patch_ops_without_doc_key(ops))
    except Exception as e:
        # Deliberately broad: this patch is model-generated, so a path that
        # doesn't exist in the actual document (jsonpointer.JsonPointerException),
        # a malformed op (jsonpatch.JsonPatchException/InvalidJsonPatch), or
        # anything else in between is an expected outcome for a wrong
        # prediction, not a bug -- e.g. a path into the wrong document of a
        # multi-document file. finetune/evaluate.py's apply_multidoc_patch
        # uses this same broad catch for the same reason.
        # It must never propagate into the UI as an unhandled exception.
        return None, f"patch failed to apply: {e}"

    return yaml.safe_dump_all(docs, sort_keys=False, default_flow_style=False, allow_unicode=True), None


def unmatched_patch_ops(findings: list[dict], patch_ops: list[dict]) -> list[dict]:
    """Patch ops that don't relate to any finding (rare, but a UI offering
    per-finding selection needs somewhere to put them)."""
    matched_ids = {id(op) for f in findings for op in patch_ops_for_finding(f, patch_ops)}
    return [op for op in patch_ops if id(op) not in matched_ids]
