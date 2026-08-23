#!/usr/bin/env bash
# Compresses large dataset/generation .jsonl files (both the single-defect
# and multi-defect strategy outputs -- see README.md's "Dataset generation
# strategies" section) so they fit under GitHub's 100MiB file limit
# (dataset/output/unsloth/train.jsonl alone is ~130MB uncompressed -- GitHub
# rejects it outright). Run this before committing. The raw .jsonl files are
# gitignored; only the .jsonl.gz files get committed. Run
# scripts/decompress_dataset.sh after cloning/pulling to get the raw files
# back.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

shopt -s nullglob
files=(
    dataset/output/*.jsonl
    dataset/output/unsloth/*.jsonl
    dataset/output-multi-defect/*.jsonl
    dataset/output-multi-defect/unsloth/*.jsonl
    generation/output/*.jsonl
)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No .jsonl files found to compress."
    exit 0
fi

for f in "${files[@]}"; do
    echo "Compressing $f"
    gzip -kf "$f"
done

echo "Done. Commit the .jsonl.gz files -- the .jsonl originals are gitignored."
