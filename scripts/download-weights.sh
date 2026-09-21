#!/usr/bin/env bash
# Download model weights into ./models via a throwaway HF container.
# No HF tooling needed on the host. Called by ./setup.sh.
#
# Usage: scripts/download-weights.sh [--force]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_docker

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if weights_present && [[ $FORCE -eq 0 ]]; then
  log "weights already present at ${MODEL_DIR} (use --force to re-download)"
  exit 0
fi

mkdir -p "${MODEL_DIR}"

log "downloading ${MODEL_REPO} -> ${MODEL_DIR}"

# --user keeps the downloaded files owned by the caller, not root.
# allow_patterns mirrors laya.Agent's own filter, so we fetch exactly what load() reads
# and skip sibling checkpoints bundled in the same repo.
# HF_TOKEN is passed through only if set (needed for gated repos); never baked in.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "${MODEL_DIR}:/out" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  --env HOME=/tmp \
  python:3.12-slim \
  bash -c '
    set -euo pipefail
    pip install --quiet --no-cache-dir "huggingface_hub[hf_transfer]>=0.20.0"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "'"${MODEL_REPO}"'",
    local_dir="/out",
    token=os.environ.get("HF_TOKEN") or None,
    allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
)
print("download complete")
PY
  '

weights_present || die "download finished but ${MODEL_DIR} is missing expected files"

log "weights ready: $(du -sh "${MODEL_DIR}" | cut -f1) at ${MODEL_DIR}"
