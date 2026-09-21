#!/usr/bin/env bash
# Start (or resume) the laya container.

source "$(dirname "${BASH_SOURCE[0]}")/scripts/lib.sh"

require_docker

docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 \
  || die "image ${IMAGE_NAME} not found -- run ./setup.sh first"
weights_present \
  || die "weights missing at ${MODEL_DIR} -- run ./setup.sh first"

if container_running; then
  log "already running: ${CONTAINER_NAME}"
  exit 0
fi

log "starting ${CONTAINER_NAME}"
compose up --detach

container_running || die "container failed to start -- check: docker logs ${CONTAINER_NAME}"

log "running. next: scripts/smoke-test.sh"
