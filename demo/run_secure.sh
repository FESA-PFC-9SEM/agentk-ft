#!/usr/bin/env bash
# Starts the secured demo: resets the demo workloads to their baseline,
# then serves the Streamlit app over HTTPS with Keycloak login, running
# kubectl as the agent's restricted service account.
#
# One-time setup first:  finetune/.venv/bin/python deploy/setup_secure_demo.py
#
# Usage:
#   demo/run_secure.sh                 # reset the cluster, then start
#   demo/run_secure.sh --no-reset      # keep the cluster as it is
#   demo/run_secure.sh --tools mcp     # the agent's tools run in an MCP server
#                                      # (demo/mcp_server.py) instead of in-process
#   --tools function (the default) runs them in-process (function calling).
set -euo pipefail

RESET=1
TOOLS=function
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-reset) RESET=0 ;;
        --tools) TOOLS="${2:-}"; shift ;;
        --tools=*) TOOLS="${1#*=}" ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
[[ "$TOOLS" == function || "$TOOLS" == mcp ]] || { echo "--tools must be 'function' or 'mcp', not '$TOOLS'" >&2; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SECRETS=deploy/.secrets
for f in "$SECRETS/app.crt" "$SECRETS/app.key" "$SECRETS/ca-bundle.pem" "$SECRETS/agent.kubeconfig" .streamlit/secrets.toml; do
    [[ -f "$f" ]] || { echo "missing $f -- run: finetune/.venv/bin/python deploy/setup_secure_demo.py" >&2; exit 1; }
done

if [[ "$RESET" == 1 ]]; then
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

# The agent's tool layer: function calling (in-process) or the MCP server.
export DEMO_TOOL_LAYER="$TOOLS"

echo ">>> starting https://localhost:8501 (tool layer: $TOOLS)"
exec finetune/.venv/bin/streamlit run demo/streamlit_app.py \
    --server.address localhost \
    --server.sslCertFile "$SECRETS/app.crt" \
    --server.sslKeyFile "$SECRETS/app.key"
