"""
Resets the demo's workloads to a saved baseline, so every demo starts from
the same flawed state -- whatever the agent fixed last time.

  python demo/reset_cluster.py --snapshot   # save the current state as the baseline (once)
  python demo/reset_cluster.py              # restore it (demo/run_secure.sh does this on start)

The baseline is one YAML file per managed namespace in demo/baseline/: the
top-level workloads, services and configmaps, stripped of everything the
cluster fills in (status, uids, cluster IPs, injected token volumes...).
Resetting deletes exactly those objects and recreates them -- deletion,
not `apply`, because a Pod's spec can't be patched back -- and removes every
resource the agent created (they carry core.AGENT_LABEL, e.g. the Secrets a
credential fix adds). Secrets are never written to the baseline.

Only the managed namespaces (core.MANAGED_NAMESPACES) are touched: never
keycloak, agent-system or the system namespaces. Runs with your own kubectl
credentials, not the agent's restricted ones.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.core import AGENT_LABEL, MANAGED_NAMESPACES, PROTECTED_NAMESPACES  # noqa: E402

BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
KINDS = "deployments,statefulsets,daemonsets,replicationcontrollers,jobs,cronjobs,pods,services,configmaps"
_DROP_ANNOTATIONS = {"kubectl.kubernetes.io/last-applied-configuration", "deployment.kubernetes.io/revision"}
_JOB_LABELS = {"controller-uid", "job-name", "batch.kubernetes.io/controller-uid", "batch.kubernetes.io/job-name"}


def admin_kubectl(*args: str, input_text: str | None = None, check: bool = True) -> str:
    result = subprocess.run(["kubectl", *args], capture_output=True, text=True, input=input_text)
    if check and result.returncode != 0:
        raise SystemExit(f"kubectl {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
    return result.stdout


def clean(obj: dict) -> dict | None:
    """The object as it should be recreated, or None if it isn't part of the
    baseline (owned by another object, created by the agent, or one the
    cluster manages itself)."""
    meta = obj.get("metadata") or {}
    labels = meta.get("labels") or {}
    if meta.get("ownerReferences") or labels.get(AGENT_LABEL[0]) == AGENT_LABEL[1]:
        return None
    kind, name = obj.get("kind"), meta.get("name")
    if (kind == "Service" and name == "kubernetes") or (kind == "ConfigMap" and name == "kube-root-ca.crt"):
        return None

    out = {"apiVersion": obj["apiVersion"], "kind": kind, "metadata": {"name": name}}
    if labels:
        out["metadata"]["labels"] = dict(labels)
    annotations = {k: v for k, v in (meta.get("annotations") or {}).items() if k not in _DROP_ANNOTATIONS}
    if annotations:
        out["metadata"]["annotations"] = annotations
    for key in ("spec", "data", "binaryData"):
        if key in obj:
            out[key] = json.loads(json.dumps(obj[key]))

    spec = out.get("spec") or {}
    if kind == "Service":
        for key in ("clusterIP", "clusterIPs", "ipFamilies", "ipFamilyPolicy"):
            spec.pop(key, None)
    pod_spec = spec if kind == "Pod" else None
    if kind == "Job":
        spec.pop("selector", None)
        tmpl_labels = ((spec.get("template") or {}).get("metadata") or {}).get("labels") or {}
        for key in _JOB_LABELS:
            tmpl_labels.pop(key, None)
        for key in _JOB_LABELS:
            out["metadata"].get("labels", {}).pop(key, None)
        pod_spec = (spec.get("template") or {}).get("spec")
    if pod_spec is not None:
        pod_spec.pop("nodeName", None)
        injected = {v["name"] for v in pod_spec.get("volumes") or [] if str(v.get("name", "")).startswith("kube-api-access-")}
        if injected:
            pod_spec["volumes"] = [v for v in pod_spec["volumes"] if v["name"] not in injected]
            if not pod_spec["volumes"]:
                pod_spec.pop("volumes")
            for container in (pod_spec.get("containers") or []) + (pod_spec.get("initContainers") or []):
                mounts = [m for m in container.get("volumeMounts") or [] if m.get("name") not in injected]
                if mounts:
                    container["volumeMounts"] = mounts
                else:
                    container.pop("volumeMounts", None)
    if out["metadata"].get("labels") == {}:
        out["metadata"].pop("labels")
    return out


def managed_namespaces() -> list[str]:
    namespaces = [ns for ns in MANAGED_NAMESPACES if ns not in PROTECTED_NAMESPACES]
    if len(namespaces) != len(MANAGED_NAMESPACES):
        raise SystemExit("a protected namespace is listed as managed -- refusing")
    return namespaces


def snapshot() -> None:
    BASELINE_DIR.mkdir(exist_ok=True)
    for namespace in managed_namespaces():
        items = json.loads(admin_kubectl("get", KINDS, "-n", namespace, "-o", "json"))["items"]
        objects = [o for o in (clean(item) for item in items) if o is not None]
        path = BASELINE_DIR / f"{namespace}.yaml"
        path.write_text(
            f"# Baseline for namespace '{namespace}' -- written by demo/reset_cluster.py --snapshot.\n"
            + yaml.safe_dump_all(objects, sort_keys=False)
        )
        print(f"{namespace}: {len(objects)} object(s) -> {path.relative_to(BASELINE_DIR.parent.parent)}")


def reset() -> None:
    namespaces = managed_namespaces()
    missing = [ns for ns in namespaces if not (BASELINE_DIR / f"{ns}.yaml").exists()]
    if missing:
        raise SystemExit(f"no baseline for {', '.join(missing)}: run `python demo/reset_cluster.py --snapshot` first")
    for namespace in namespaces:
        path = str(BASELINE_DIR / f"{namespace}.yaml")
        label = f"{AGENT_LABEL[0]}={AGENT_LABEL[1]}"
        admin_kubectl("delete", "secrets,configmaps", "-n", namespace, "-l", label, "--ignore-not-found")
        admin_kubectl("delete", "-f", path, "-n", namespace, "--ignore-not-found", "--wait=true", "--timeout=120s")
        admin_kubectl("apply", "-f", path, "-n", namespace)
        print(f"{namespace}: restored to the baseline")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", action="store_true", help="save the current state as the baseline instead of restoring it")
    args = parser.parse_args(argv)
    snapshot() if args.snapshot else reset()


if __name__ == "__main__":
    main()
