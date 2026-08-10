#!/usr/bin/env bash
# Chains the full pipeline: generate synthetic manifests via the local model ->
# curate -> merge with the real corpus -> run dataset/build.py -> final
# dataset.jsonl with balanced per-rule quotas and repo-grouped split.
#
# Usage:
#   ./pipeline.sh                       # full run (production defaults)
#   ./pipeline.sh --smoke               # small subset, to validate the pipeline end to end
#
# Configurable environment variables (with defaults):
#   CORPUS_DIR, OUTPUT_DIR, SEED, N_BASE, N_HARD_NEGATIVE, TOTAL,
#   LIMIT (real corpus rows read; empty = no limit), CONCURRENCY
#
# Note: RBAC generation (--mode rbac) is intentionally not run here. It exists
# to feed KSEC-004, which is currently disabled (see README.md's "Active vs.
# disabled rules") -- generating it would just be wasted GPU time until
# KSEC-004 is re-enabled. Run `generation.generate --mode rbac` manually if
# you need it for something else.

set -euo pipefail

export PATH="$PATH:$(go env GOPATH 2>/dev/null)/bin"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python"

CORPUS_DIR="${CORPUS_DIR:-corpus}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/output}"
GENERATION_DIR="generation/output"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-4}"

if [[ "${1:-}" == "--smoke" ]]; then
    echo ">>> Smoke mode: small subset to validate the pipeline end to end"
    N_BASE="${N_BASE:-15}"
    N_HARD_NEGATIVE="${N_HARD_NEGATIVE:-15}"
    TOTAL="${TOTAL:-150}"
    LIMIT="${LIMIT:-1000}"
else
    N_BASE="${N_BASE:-3000}"
    N_HARD_NEGATIVE="${N_HARD_NEGATIVE:-1500}"
    TOTAL="${TOTAL:-20000}"
    LIMIT="${LIMIT:-}"
fi

echo ">>> Step 1/4: synthetic generation (base=${N_BASE}, hard-negative=${N_HARD_NEGATIVE})"
$PYTHON -m generation.generate --mode base --seed "$SEED" -n "$N_BASE" --concurrency "$CONCURRENCY"
$PYTHON -m generation.generate --mode hard-negative --seed "$SEED" -n "$N_HARD_NEGATIVE" --concurrency "$CONCURRENCY"

echo ">>> Step 2/4: curation"
$PYTHON -m generation.curate --mode base
$PYTHON -m generation.curate --mode hard-negative

echo ">>> Step 2b/4: diversity report"
for mode in base hard-negative; do
    curated="$GENERATION_DIR/${mode}.curated.jsonl"
    if [[ -s "$curated" ]]; then
        $PYTHON -m generation.report --input "$curated" --output-dir "$GENERATION_DIR/report-${mode}"
    fi
done

echo ">>> Step 3-4/4: build.py (merges real + synthetic corpus, injects defects, writes final dataset)"
BUILD_ARGS=(--corpus-dir "$CORPUS_DIR" --synthetic-dir "$GENERATION_DIR" --output-dir "$OUTPUT_DIR" --seed "$SEED" --total "$TOTAL")
if [[ -n "$LIMIT" ]]; then
    BUILD_ARGS+=(--limit "$LIMIT")
fi
$PYTHON -m dataset.build "${BUILD_ARGS[@]}"

echo ">>> Pipeline complete. Splits at ${OUTPUT_DIR}/{train,val,test}.jsonl"
