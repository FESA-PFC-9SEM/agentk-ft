"""
One-time (and re-runnable) setup for the secured demo, on the local minikube:

  1. a local certificate authority and two HTTPS certificates signed by it --
     Keycloak (https://<minikube ip>:30443) and the Streamlit app
     (https://localhost:8501);
  2. Keycloak in its own `keycloak` namespace (deploy/keycloak.yaml), with a
     realm imported at startup: the app's OIDC client, two roles -- viewer
     (chat, list, audit) and operator (may also confirm cluster changes) --
     and one demo user per role;
  3. the agent's own service account with RBAC (deploy/agent-rbac.yaml):
     cluster-wide read without Secrets, writes only in the managed
     namespaces -- never in `keycloak` -- and a kubeconfig for it;
  4. .streamlit/secrets.toml with the app's login settings.

Everything generated (keys, passwords, the client secret, the kubeconfig)
stays in deploy/.secrets/ and .streamlit/secrets.toml, both git-ignored.
Re-running reuses the stored passwords and keys, so it's safe to run again
(e.g. after `minikube delete`).

Usage:
    finetune/.venv/bin/python deploy/setup_secure_demo.py
    demo/run_secure.sh
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SECRETS_DIR = REPO_ROOT / "deploy" / ".secrets"
STREAMLIT_SECRETS = REPO_ROOT / ".streamlit" / "secrets.toml"
REALM = "k8s-agent"
CLIENT_ID = "cluster-agent-ui"
APP_URL = "https://localhost:8501"
KEYCLOAK_NODE_PORT = 30443


def run(*args: str, input_text: str | None = None, check: bool = True) -> str:
    result = subprocess.run(list(args), capture_output=True, text=True, input=input_text)
    if check and result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(args)}\n{result.stderr or result.stdout}")
    return result.stdout


def kubectl_apply(manifest: str) -> None:
    print(run("kubectl", "apply", "-f", "-", input_text=manifest).strip())


# ---------------------------------------------------------------------------
# Secrets and certificates
# ---------------------------------------------------------------------------


def load_or_create_values() -> dict:
    """Passwords and secrets, generated once and reused."""
    path = SECRETS_DIR / "values.json"
    values = json.loads(path.read_text()) if path.exists() else {}
    for key in ("admin_password", "client_secret", "viewer_password", "operator_password", "cookie_secret"):
        values.setdefault(key, secrets.token_urlsafe(24))
    path.write_text(json.dumps(values, indent=2))
    path.chmod(0o600)
    return values


def make_certificates(minikube_ip: str) -> None:
    """A local CA, then a Keycloak and an app certificate it signs. The CA is
    what Python (and, if you import it, your browser) trusts."""
    ca_key, ca_crt = SECRETS_DIR / "ca.key", SECRETS_DIR / "ca.crt"
    if not ca_crt.exists():
        run("openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes", "-days", "3650",
            "-keyout", str(ca_key), "-out", str(ca_crt), "-subj", "/CN=Cluster Agent Demo Local CA")
    for name, sans in (
        ("keycloak", f"IP:{minikube_ip},DNS:keycloak.keycloak.svc,DNS:localhost"),
        ("app", "DNS:localhost,IP:127.0.0.1"),
    ):
        key, csr, crt = (SECRETS_DIR / f"{name}.{ext}" for ext in ("key", "csr", "crt"))
        if crt.exists() and sans.split(",")[0].split(":", 1)[1] in run("openssl", "x509", "-in", str(crt), "-noout", "-ext", "subjectAltName"):
            continue
        run("openssl", "req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(csr), "-subj", f"/CN={name}")
        ext = SECRETS_DIR / f"{name}.ext"
        ext.write_text(f"subjectAltName={sans}\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n")
        run("openssl", "x509", "-req", "-in", str(csr), "-CA", str(ca_crt), "-CAkey", str(ca_key), "-CAcreateserial",
            "-days", "825", "-sha256", "-out", str(crt), "-extfile", str(ext))
    for path in SECRETS_DIR.glob("*.key"):
        path.chmod(0o600)
    # Python's trust store for the app process: the usual public CAs plus the
    # local one, so other HTTPS calls keep working.
    try:
        import certifi

        public = Path(certifi.where()).read_text()
    except ImportError:
        public = ""
    (SECRETS_DIR / "ca-bundle.pem").write_text(public + "\n" + ca_crt.read_text())


# ---------------------------------------------------------------------------
# Keycloak
# ---------------------------------------------------------------------------


def realm_json(values: dict) -> str:
    def user(name: str, roles: list[str]) -> dict:
        return {
            "username": name,
            "enabled": True,
            "email": f"{name}@cluster-agent.local",
            "emailVerified": True,
            "firstName": name.capitalize(),
            "lastName": "Demo",
            "credentials": [{"type": "password", "value": values[f"{name}_password"], "temporary": False}],
            "realmRoles": roles,
        }

    realm = {
        "realm": REALM,
        "enabled": True,
        "sslRequired": "all",
        "roles": {
            "realm": [
                {"name": "viewer", "description": "Chat with the agent, list workloads, audit manifests."},
                {"name": "operator", "description": "Everything a viewer can do, plus confirm changes to the cluster."},
            ]
        },
        "clients": [
            {
                "clientId": CLIENT_ID,
                "name": "Kubernetes Cluster Agent",
                "enabled": True,
                "protocol": "openid-connect",
                "publicClient": False,
                "clientAuthenticatorType": "client-secret",
                "secret": values["client_secret"],
                "standardFlowEnabled": True,
                "directAccessGrantsEnabled": False,
                "redirectUris": [f"{APP_URL}/oauth2callback"],
                "webOrigins": [APP_URL],
                "attributes": {"post.logout.redirect.uris": f"{APP_URL}/*"},
                # Realm roles as a "roles" claim in the ID token, which is
                # what Streamlit's st.user exposes.
                "protocolMappers": [
                    {
                        "name": "realm roles",
                        "protocol": "openid-connect",
                        "protocolMapper": "oidc-usermodel-realm-role-mapper",
                        "config": {
                            "claim.name": "roles",
                            "jsonType.label": "String",
                            "multivalued": "true",
                            "id.token.claim": "true",
                            "access.token.claim": "true",
                            "userinfo.token.claim": "true",
                        },
                    }
                ],
            }
        ],
        "users": [user("viewer", ["viewer"]), user("operator", ["viewer", "operator"])],
    }
    return json.dumps(realm, indent=2)


def secret_manifest(name: str, namespace: str, data: dict, secret_type: str = "Opaque") -> str:
    encoded = {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}
    return json.dumps({"apiVersion": "v1", "kind": "Secret", "metadata": {"name": name, "namespace": namespace}, "type": secret_type, "data": encoded})


def deploy_keycloak(values: dict, keycloak_url: str) -> None:
    kubectl_apply(json.dumps({"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "keycloak"}}))
    kubectl_apply(secret_manifest("keycloak-admin", "keycloak", {"username": "admin", "password": values["admin_password"]}))
    kubectl_apply(secret_manifest(
        "keycloak-tls", "keycloak",
        {"tls.crt": (SECRETS_DIR / "keycloak.crt").read_text(), "tls.key": (SECRETS_DIR / "keycloak.key").read_text()},
        "kubernetes.io/tls",
    ))
    kubectl_apply(secret_manifest("keycloak-realm", "keycloak", {"realm.json": realm_json(values)}))
    kubectl_apply(json.dumps({
        "apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "keycloak-config", "namespace": "keycloak"},
        "data": {"hostname": keycloak_url},
    }))
    kubectl_apply((REPO_ROOT / "deploy" / "keycloak.yaml").read_text())
    # Secrets/realm may have changed: restart so Keycloak re-imports them.
    run("kubectl", "rollout", "restart", "deployment/keycloak", "-n", "keycloak")


def wait_for_keycloak(keycloak_url: str, timeout: int = 300) -> None:
    import ssl
    import urllib.request

    context = ssl.create_default_context(cafile=str(SECRETS_DIR / "ca.crt"))
    url = f"{keycloak_url}/realms/{REALM}/.well-known/openid-configuration"
    deadline = time.time() + timeout
    run("kubectl", "rollout", "status", "deployment/keycloak", "-n", "keycloak", f"--timeout={timeout}s")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, context=context, timeout=5) as response:
                issuer = json.load(response)["issuer"]
                print(f"Keycloak is up, realm issuer: {issuer}")
                return
        except Exception as e:  # noqa: BLE001 -- still starting
            last = e
            time.sleep(5)
    raise SystemExit(f"Keycloak didn't answer at {url}: {last}")


# ---------------------------------------------------------------------------
# The agent's identity
# ---------------------------------------------------------------------------


def deploy_agent_rbac(namespaces: list[str]) -> None:
    from demo.core import PROTECTED_NAMESPACES

    kubectl_apply((REPO_ROOT / "deploy" / "agent-rbac.yaml").read_text())
    for namespace in namespaces:
        if namespace in PROTECTED_NAMESPACES:
            raise SystemExit(f"'{namespace}' is protected and can't be a managed namespace")
        kubectl_apply(json.dumps({"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}}))
        kubectl_apply(json.dumps({
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {"name": "kube-agent-write", "namespace": namespace},
            "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "kube-agent-write"},
            "subjects": [{"kind": "ServiceAccount", "name": "kube-agent", "namespace": "agent-system"}],
        }))


def write_agent_kubeconfig() -> Path:
    for _ in range(30):
        token_b64 = run("kubectl", "get", "secret", "kube-agent-token", "-n", "agent-system", "-o", "jsonpath={.data.token}", check=False)
        if token_b64:
            break
        time.sleep(1)
    else:
        raise SystemExit("the kube-agent token wasn't issued")
    current = json.loads(run("kubectl", "config", "view", "--raw", "--minify", "-o", "json"))
    cluster = current["clusters"][0]["cluster"]
    ca_data = cluster.get("certificate-authority-data") or base64.b64encode(Path(cluster["certificate-authority"]).read_bytes()).decode()
    kubeconfig = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": "minikube", "cluster": {"server": cluster["server"], "certificate-authority-data": ca_data}}],
        "users": [{"name": "kube-agent", "user": {"token": base64.b64decode(token_b64).decode()}}],
        "contexts": [{"name": "kube-agent", "context": {"cluster": "minikube", "user": "kube-agent"}}],
        "current-context": "kube-agent",
    }
    path = SECRETS_DIR / "agent.kubeconfig"
    path.write_text(json.dumps(kubeconfig, indent=2))
    path.chmod(0o600)
    return path


def verify_agent_permissions(kubeconfig: Path, namespaces: list[str]) -> None:
    """Ask the API server itself what the agent may do."""
    def can(verb: str, resource: str, namespace: str) -> bool:
        return run("kubectl", "--kubeconfig", str(kubeconfig), "auth", "can-i", verb, resource, "-n", namespace, check=False).strip() == "yes"

    checks = [(f"patch deployments in {ns}", can("patch", "deployments", ns), True) for ns in namespaces]
    checks += [
        ("list pods in all namespaces", run("kubectl", "--kubeconfig", str(kubeconfig), "auth", "can-i", "list", "pods", "-A", check=False).strip() == "yes", True),
        ("patch deployments in keycloak", can("patch", "deployments", "keycloak"), False),
        ("get secrets in keycloak", can("get", "secrets", "keycloak"), False),
        ("delete deployments in demo", can("delete", "deployments", "demo"), False),
    ]
    for label, allowed, expected in checks:
        print(f"  agent can {label}: {'yes' if allowed else 'no'}" + ("" if allowed == expected else "   <-- UNEXPECTED"))
        if allowed != expected:
            raise SystemExit("the agent's RBAC isn't what it should be")


# ---------------------------------------------------------------------------


def write_streamlit_secrets(values: dict, keycloak_url: str) -> None:
    STREAMLIT_SECRETS.write_text(
        "# Generated by deploy/setup_secure_demo.py -- git-ignored, do not commit.\n"
        "[auth]\n"
        f'redirect_uri = "{APP_URL}/oauth2callback"\n'
        f'cookie_secret = "{values["cookie_secret"]}"\n'
        f'client_id = "{CLIENT_ID}"\n'
        f'client_secret = "{values["client_secret"]}"\n'
        f'server_metadata_url = "{keycloak_url}/realms/{REALM}/.well-known/openid-configuration"\n'
    )
    STREAMLIT_SECRETS.chmod(0o600)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    from demo.core import MANAGED_NAMESPACES

    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    SECRETS_DIR.chmod(0o700)
    minikube_ip = run("minikube", "ip").strip()
    keycloak_url = f"https://{minikube_ip}:{KEYCLOAK_NODE_PORT}"

    print(">>> secrets and certificates")
    values = load_or_create_values()
    make_certificates(minikube_ip)
    print(">>> Keycloak")
    deploy_keycloak(values, keycloak_url)
    wait_for_keycloak(keycloak_url)
    print(">>> the agent's service account and RBAC")
    deploy_agent_rbac(MANAGED_NAMESPACES)
    kubeconfig = write_agent_kubeconfig()
    verify_agent_permissions(kubeconfig, MANAGED_NAMESPACES)
    print(">>> app login settings")
    write_streamlit_secrets(values, keycloak_url)

    print(f"""
Done.
  App:       {APP_URL}            (start it with: demo/run_secure.sh)
  Keycloak:  {keycloak_url}/admin   admin / {values['admin_password']}
  Users:     viewer   / {values['viewer_password']}
             operator / {values['operator_password']}
Your browser will warn about the certificate once; to avoid it, import
{SECRETS_DIR / 'ca.crt'} as a trusted authority.""")


if __name__ == "__main__":
    main()
