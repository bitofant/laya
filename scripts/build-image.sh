#!/usr/bin/env bash
# Build the Laya runtime image. Called by ./setup.sh; safe to run directly to rebuild.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_docker

log "building ${IMAGE_NAME}"
docker build \
  --tag "${IMAGE_NAME}" \
  "${@}" \
  "${REPO_ROOT}/docker"

log "built ${IMAGE_NAME}"
