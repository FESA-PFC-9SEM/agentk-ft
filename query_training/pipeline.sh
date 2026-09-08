#!/usr/bin/env bash
# Builds the kubectl-query training data end to end from the raw corpora under
# query_training/corpus/, then (optionally) trains, plots and evaluates.
#
# This task is separate from the manifest-security pipeline in ../pipeline.sh.
#
# Usage:
#   query_training/pipeline.sh                      # clean + export only (no GPU needed)
#   query_training/pipeline.sh --train              # also train + plot + evaluate (needs finetune/.venv + GPU)
#   TARGET_FORMAT=plan-command query_training/pipeline.sh --train
#   SOURCES=kubectl-command-csv,devops-kubectl query_training/pipeline.sh
#   READ_ONLY=1 query_training/pipeline.sh
#
# Environment variables (with defaults):
#   SOURCES       (kubectl-command-csv)   comma-separated; also: devops-kubectl
#   TARGET_FORMAT (command)               or plan-command ({"plan","command"} JSON target)
#   ON_CONFLICT   (first)                 one-instruction-many-commands policy
#   READ_ONLY     (unset)                 set to 1 to drop mutating verbs
#   TOKENIZER     (unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit)
#   MODEL         (same)                  base model for --train
#   MAIN_PY       (.venv/bin/python)
#   TRAIN_PY      (finetune/.venv/bin/python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

MAIN_PY="${MAIN_PY:-.venv/bin/python}"
TRAIN_PY="${TRAIN_PY:-finetune/.venv/bin/python}"
SOURCES="${SOURCES:-kubectl-command-csv}"
TARGET_FORMAT="${TARGET_FORMAT:-command}"
ON_CONFLICT="${ON_CONFLICT:-first}"
TOKENIZER="${TOKENIZER:-unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit}"
MODEL="${MODEL:-$TOKENIZER}"

CLEAN_ARGS=(--sources "$SOURCES" --target-format "$TARGET_FORMAT" --on-conflict "$ON_CONFLICT")
if [[ -n "${READ_ONLY:-}" ]]; then
    CLEAN_ARGS+=(--read-only)
fi

echo ">>> Step 1/2: clean [$SOURCES] -> query_training/output/{train,val,test}.jsonl (target=$TARGET_FORMAT)"
$MAIN_PY -m query_training.clean "${CLEAN_ARGS[@]}"

echo ">>> Step 2/2: render chat template + count tokens -> query_training/output/unsloth/"
$MAIN_PY -m query_training.export --tokenizer "$TOKENIZER"

if [[ "${1:-}" != "--train" ]]; then
    echo ">>> Done (data only). Pass --train to also train + evaluate."
    exit 0
fi

echo ">>> Training ($MODEL)"
$TRAIN_PY -m query_training.train --model "$MODEL"

RUN_DIR="$(ls -dt query_training/runs/*/ | head -1)"

echo ">>> Plotting ${RUN_DIR}metrics.jsonl"
$MAIN_PY -m query_training.plot_metrics --metrics-file "${RUN_DIR}metrics.jsonl"

echo ">>> Evaluating ${RUN_DIR}lora_adapter"
$TRAIN_PY -m query_training.evaluate --adapter "${RUN_DIR}lora_adapter"

echo ">>> Pipeline complete. Run: ${RUN_DIR}"
