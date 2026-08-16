#!/usr/bin/env bash
# Inflates committed .jsonl.gz files back into the .jsonl files that
# dataset/build.py, finetune/train_unsloth.py, etc. expect. Run this after
# cloning/pulling on a fresh machine (e.g. a training VM), before training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

shopt -s nullglob
files=(dataset/output/*.jsonl.gz dataset/output/unsloth/*.jsonl.gz generation/output/*.jsonl.gz)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No .jsonl.gz files found to decompress."
    exit 0
fi

for f in "${files[@]}"; do
    echo "Decompressing $f"
    gzip -dkf "$f"
done

echo "Done."
