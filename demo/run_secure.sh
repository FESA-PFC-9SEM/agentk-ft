#!/usr/bin/env bash
# Starts the secured demo: resets the demo workloads to their baseline,
# then serves the Streamlit app over HTTPS with Keycloak login, running
# kubectl as the agent's restricted service account.
#
# One-time setup first:  finetune/.venv/bin/python deploy/setup_secure_demo.py
#
# Usage:
#   demo/run_secure.sh             # reset the cluster, then start
#   demo/run_secure.sh --no-reset  # keep the cluster as it is
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SECRETS=deploy/.secrets
for f in "$SECRETS/app.crt" "$SECRETS/app.key" "$SECRETS/ca-bundle.pem" "$SECRETS/agent.kubeconfig" .streamlit/secrets.toml; do
    [[ -f "$f" ]] || { echo "missing $f -- run: finetune/.venv/bin/python deploy/setup_secure_demo.py" >&2; exit 1; }
done

if [[ "${1:-}" != "--no-reset" ]]; then
    echo ">>> resetting demo workloads to the baseline (demo/baseline/)"
    finetune/.venv/bin/python demo/reset_cluster.py
fi

# The agent's kubectl calls use its own service account (RBAC: no access to
# the keycloak namespace, no deletes); Python trusts the local CA, so the
# app can talk to Keycloak over HTTPS.
export AGENT_KUBECONFIG="$PWD/$SECRETS/agent.kubeconfig"
# The model the sidebar selects by default (override: DEMO_ADAPTER=runs/<run>/lora_adapter).
export DEMO_ADAPTER="${DEMO_ADAPTER:-runs/20261003-0240_multi-defect_qwen2_5-coder-7b-instruct-bnb-4bit/lora_adapter}"
export SSL_CERT_FILE="$PWD/$SECRETS/ca-bundle.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

echo ">>> starting https://localhost:8501"
exec finetune/.venv/bin/streamlit run demo/streamlit_app.py \
    --server.address localhost \
    --server.sslCertFile "$SECRETS/app.crt" \
    --server.sslKeyFile "$SECRETS/app.key"
