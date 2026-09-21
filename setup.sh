#!/usr/bin/env bash
# One-time setup: check prerequisites, build the image, download model weights.
# Idempotent -- safe to re-run. Use --force to re-download weights.

source "$(dirname "${BASH_SOURCE[0]}")/scripts/lib.sh"

log "checking prerequisites"
require_docker
require_gpu
log "docker + GPU passthrough ok: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

"${REPO_ROOT}/scripts/build-image.sh"
"${REPO_ROOT}/scripts/download-weights.sh" "$@"

cat <<EOF

setup complete.

  ./start.sh              start the laya container
  scripts/smoke-test.sh   verify GPU + inference
  ./stop.sh               stop it again
EOF
