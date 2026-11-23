"""
Runs the whole demo inside Kubernetes: the app (Streamlit UI, planner,
fine-tuned auditor, MCP server) becomes a pod next to Keycloak and
PostgreSQL, instead of a process on the host.

Prerequisites:
  1. a minikube cluster whose node exposes the GPU:
         minikube start --gpus all
     (--gpus only applies when the cluster is created: an existing cluster
     has to be deleted and recreated -- see demo/README.md);
  2. deploy/setup_secure_demo.py has run against that cluster (Keycloak,
     PostgreSQL, the agent's service account, certificates, passwords).

What this script does (re-runnable):
  1. checks both prerequisites;
  2. builds the image inside minikube (deploy/app/Dockerfile) from a staged
     context: the app's packages and the one adapter it serves;
  3. creates the app's Secrets (HTTPS certificate, the local CA, the
     Keycloak login and the audit database URL -- in-cluster addresses);
  4. applies deploy/agent-app.yaml with the chosen tool layer and adapter,
     and the reset account's RoleBindings in the managed namespaces;
  5. resets the demo workloads with a Job (deploy/demo-reset-job.yaml),
     unless --no-reset;
  6. restarts the app and waits for it. The first start downloads the base
     model (~5.5 GB) into the agent-models volume.

Usage:
    finetune/.venv/bin/python deploy/deploy_app_k8s.py                  # function-calling tool layer
    finetune/.venv/bin/python deploy/deploy_app_k8s.py --tools mcp      # MCP server tool layer
    finetune/.venv/bin/python deploy/deploy_app_k8s.py --no-build --tools mcp   # switch layer, same image
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.setup_secure_demo import (  # noqa: E402
    APP_NODE_PORT,
    CLIENT_ID,
    KEYCLOAK_NODE_PORT,
    REALM,
    SECRETS_DIR,
    kubectl_apply,
    make_certificates,
    run,
    secret_manifest,
)

NAMESPACE = "agent-system"
IMAGE = "agent-app:dev"
DEFAULT_ADAPTER = "runs/20261003-0240_multi-defect_qwen2_5-coder-7b-instruct-bnb-4bit/lora_adapter"
# What the image needs from the repository -- nothing else is copied (no
# datasets, no checkpoints, never deploy/.secrets or .streamlit/secrets.toml).
STAGED = {
    "dataset": ["*.py", "name_vocab.json"],
    "finetune": ["*.py", "requirements.txt"],
    "demo": ["*.py", "requirements.txt", "static/*"],
    "deploy/app": ["*"],
    ".streamlit": ["config.toml"],
}


def check_prerequisites() -> str:
    """The minikube IP, once the GPU and the secure setup are confirmed."""
    if not (SECRETS_DIR / "values.json").exists():
        raise SystemExit("run deploy/setup_secure_demo.py first (Keycloak, PostgreSQL, the agent's account)")
    gpus = run("kubectl", "get", "nodes", "-o", "jsonpath={.items[*].status.allocatable.nvidia\\.com/gpu}").split()
    if not any(g not in ("", "0") for g in gpus):
        raise SystemExit(
            "no node exposes an NVIDIA GPU (nvidia.com/gpu). minikube adds one only when the cluster is created:\n"
            "  minikube delete && minikube start --gpus all\n"
            "then re-run deploy/setup_secure_demo.py and demo/seed_minikube.sh (see demo/README.md)."
        )
    return run("minikube", "ip").strip()


def stage_context(adapter: str, target: Path) -> None:
    for directory, patterns in STAGED.items():
        for pattern in patterns:
            for source in (REPO_ROOT / directory).glob(pattern):
                if source.is_file():
                    destination = target / source.relative_to(REPO_ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    adapter_dir = REPO_ROOT / adapter
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"{adapter} isn't a LoRA adapter (no adapter_config.json)")
    shutil.copytree(adapter_dir, target / adapter)
    # The app lists only adapters whose run has a run_info.json (demo/core.py list_adapters).
    run_info = adapter_dir.parent / "run_info.json"
    if not run_info.exists():
        raise SystemExit(f"{run_info.relative_to(REPO_ROOT)} is missing: the app wouldn't list this adapter")
    shutil.copy2(run_info, target / Path(adapter).parent / "run_info.json")
    for forbidden in ("secrets.toml", ".secrets"):
        if list(target.rglob(forbidden)):
            raise SystemExit(f"refusing to build: {forbidden} ended up in the build context")


def build_image(adapter: str) -> None:
    with tempfile.TemporaryDirectory(prefix="agent-app-context-") as context:
        stage_context(adapter, Path(context))
        print(f">>> building {IMAGE} inside minikube (first build: ~10 min, downloads torch and CUDA libraries)")
        run("minikube", "image", "build", "-t", IMAGE, "-f", "deploy/app/Dockerfile", context)


def in_cluster_secrets(values: dict, minikube_ip: str) -> None:
    """The app's Secrets, with the addresses a pod uses: Keycloak at the
    minikube IP (the browser must reach the same issuer), PostgreSQL by its
    in-cluster DNS name (its certificate names it), TLS verified everywhere."""
    app_url = f"https://{minikube_ip}:{APP_NODE_PORT}"
    keycloak_url = f"https://{minikube_ip}:{KEYCLOAK_NODE_PORT}"
    database_url = (
        f"postgresql://audit_writer:{values['audit_writer_password']}@postgres.audit.svc:5432/audit"
        "?sslmode=verify-full&sslrootcert=/etc/agent-app/ca/ca.crt"
    )
    secrets_toml = (
        "# Generated by deploy/deploy_app_k8s.py -- the in-cluster app's login and audit settings.\n"
        "[auth]\n"
        f'redirect_uri = "{app_url}/oauth2callback"\n'
        f'cookie_secret = "{values["cookie_secret"]}"\n'
        f'client_id = "{CLIENT_ID}"\n'
        f'client_secret = "{values["client_secret"]}"\n'
        f'server_metadata_url = "{keycloak_url}/realms/{REALM}/.well-known/openid-configuration"\n'
        "\n[audit]\n"
        f'database_url = "{database_url}"\n'
    )
    kubectl_apply(secret_manifest("agent-app-config", NAMESPACE, {"secrets.toml": secrets_toml, "database-url": database_url}))
    kubectl_apply(secret_manifest(
        "agent-app-tls", NAMESPACE,
        {"tls.crt": (SECRETS_DIR / "app-k8s.crt").read_text(), "tls.key": (SECRETS_DIR / "app-k8s.key").read_text()},
        "kubernetes.io/tls",
    ))
    kubectl_apply(secret_manifest(
        "agent-app-ca", NAMESPACE,
        {"ca.crt": (SECRETS_DIR / "ca.crt").read_text(), "ca-bundle.pem": (SECRETS_DIR / "ca-bundle.pem").read_text()},
    ))


def app_manifests(tools: str, adapter: str) -> str:
    """deploy/agent-app.yaml with the tool layer and adapter filled in."""
    docs = list(yaml.safe_load_all((REPO_ROOT / "deploy" / "agent-app.yaml").read_text()))
    for doc in docs:
        if doc["kind"] != "Deployment":
            continue
        pod = doc["spec"]["template"]["spec"]
        for container in pod["containers"] + pod.get("initContainers", []):
            for env in container.get("env", []):
                if env.get("value") == "ADAPTER_PLACEHOLDER":
                    env["value"] = adapter
                elif env.get("value") == "TOOLS_PLACEHOLDER":
                    env["value"] = tools
    text = yaml.safe_dump_all(docs, sort_keys=False)
    if "PLACEHOLDER" in text:
        raise SystemExit("deploy/agent-app.yaml still has a placeholder after filling it in")
    return text


def reset_rbac_and_baseline(namespaces: list[str]) -> None:
    from demo.core import PROTECTED_NAMESPACES

    for namespace in namespaces:
        if namespace in PROTECTED_NAMESPACES:
            raise SystemExit(f"'{namespace}' is protected and can't be reset")
        kubectl_apply(json.dumps({
            "apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding",
            "metadata": {"name": "demo-reset", "namespace": namespace},
            "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "demo-reset"},
            "subjects": [{"kind": "ServiceAccount", "name": "demo-reset", "namespace": NAMESPACE}],
        }))
    baseline = {p.name: p.read_text() for p in sorted((REPO_ROOT / "demo" / "baseline").glob("*.yaml"))}
    kubectl_apply(json.dumps({"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "demo-baseline", "namespace": NAMESPACE},
                              "data": baseline}))


def run_reset_job() -> None:
    print(">>> resetting the demo workloads (Job demo-reset)")
    run("kubectl", "delete", "job", "demo-reset", "-n", NAMESPACE, "--ignore-not-found", "--wait=true")
    kubectl_apply((REPO_ROOT / "deploy" / "demo-reset-job.yaml").read_text())
    done = run("kubectl", "wait", "--for=condition=complete", "job/demo-reset", "-n", NAMESPACE, "--timeout=300s", check=False)
    print(run("kubectl", "logs", "job/demo-reset", "-n", NAMESPACE, check=False).strip())
    if "condition met" not in done:
        raise SystemExit("the reset Job didn't complete -- kubectl describe job demo-reset -n agent-system")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tools", choices=["function", "mcp"], default="function", help="the agent's tool layer")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="the LoRA adapter the app serves (baked into the image)")
    parser.add_argument("--no-build", action="store_true", help="reuse the image already in minikube")
    parser.add_argument("--no-reset", action="store_true", help="keep the demo workloads as they are")
    args = parser.parse_args(argv)
    from demo.core import MANAGED_NAMESPACES

    minikube_ip = check_prerequisites()
    values = json.loads((SECRETS_DIR / "values.json").read_text())
    make_certificates(minikube_ip)  # adds the in-cluster app's certificate if it's missing
    if not args.no_build:
        build_image(args.adapter)
    print(">>> secrets and manifests")
    in_cluster_secrets(values, minikube_ip)
    kubectl_apply(app_manifests(args.tools, args.adapter))
    reset_rbac_and_baseline(MANAGED_NAMESPACES)
    if not args.no_reset:
        run_reset_job()
    print(">>> starting the app (the first start downloads the base model: several minutes)")
    run("kubectl", "rollout", "restart", "deployment/agent-app", "-n", NAMESPACE)
    run("kubectl", "rollout", "status", "deployment/agent-app", "-n", NAMESPACE, "--timeout=1800s")
    print(f"\nDone. App: https://{minikube_ip}:{APP_NODE_PORT}  (tool layer: {args.tools})\n"
          "Log in with the users deploy/setup_secure_demo.py printed (viewer / operator).")


if __name__ == "__main__":
    main()
