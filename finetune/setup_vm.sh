#!/usr/bin/env bash
# One-command environment bring-up for a fresh training VM: installs uv (if
# missing), a Python 3.11 interpreter (this training stack does not work on
# newer Python -- see finetune/requirements.txt for why), the finetune/.venv
# with torch + the rest of the stack, and decompresses the dataset.
#
# Usage:
#   ./finetune/setup_vm.sh
#   CUDA_INDEX=https://download.pytorch.org/whl/cu128 ./finetune/setup_vm.sh   # different CUDA version
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"

if ! command -v uv >/dev/null 2>&1; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">>> Python 3.11 + finetune/.venv"
uv python install 3.11
uv venv --python 3.11 finetune/.venv

echo ">>> torch (CUDA build: ${CUDA_INDEX})"
uv pip install --python finetune/.venv/bin/python torch torchvision --index-url "$CUDA_INDEX"

echo ">>> rest of the training stack"
uv pip install --python finetune/.venv/bin/python -r finetune/requirements.txt

echo ">>> decompressing dataset"
./scripts/decompress_dataset.sh

echo
echo ">>> Setup complete. Run:"
echo "    finetune/.venv/bin/python -m finetune.train_unsloth --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
