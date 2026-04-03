#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-IDEA-Research/Rex-Omni}"
TARGET_DIR="${SCRIPT_DIR}"

echo "Downloading ${MODEL_ID} to ${TARGET_DIR}"

if command -v hf >/dev/null 2>&1; then
    hf download "${MODEL_ID}" \
        --repo-type model \
        --local-dir "${TARGET_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_ID}" \
        --repo-type model \
        --local-dir "${TARGET_DIR}"
elif python3 -m huggingface_hub --help >/dev/null 2>&1; then
    python3 -m huggingface_hub download "${MODEL_ID}" \
        --repo-type model \
        --local-dir "${TARGET_DIR}"
else
    echo "Missing Hugging Face downloader." >&2
    echo "Install one of: pip install -U \"huggingface_hub[cli]\"" >&2
    exit 1
fi

echo "Download completed: ${TARGET_DIR}"
