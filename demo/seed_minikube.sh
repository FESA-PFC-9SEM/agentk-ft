#!/usr/bin/env bash
# Applies scenarios/*.yaml (the hand-crafted demo manifests, each with a
# known planted issue -- see scenarios/test_cases.yaml for the ground truth)
# onto whatever cluster the current kubectl context points at, so
# demo/app.py's "load from live cluster" picker has real, flawed workloads to
# fetch instead of an empty/system-only minikube.
#
# Pods failing to actually start (missing images, etc.) is fine and expected
# -- the demo only needs `kubectl get -o yaml` to return the spec, not a
# Running pod.
#
# Usage:
#   minikube start
#   demo/seed_minikube.sh
#   NAMESPACE=other-ns demo/seed_minikube.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-demo}"

echo ">>> kubectl context: $(kubectl config current-context)"
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"

shopt -s nullglob
for f in "$SCRIPT_DIR"/scenarios/*.yaml; do
    base="$(basename "$f")"
    case "$base" in
        test_cases.yaml|ground_truth.yaml) continue ;;  # ground truth, not a manifest
    esac
    echo ">>> applying $base"
    kubectl apply -n "$NAMESPACE" -f "$f" || echo "    (non-fatal: $base failed to apply, continuing)"
done

echo
echo ">>> workloads now in namespace '$NAMESPACE':"
kubectl get pod,deployment,statefulset,daemonset,replicaset,job,cronjob -n "$NAMESPACE"
