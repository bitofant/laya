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

# The model loads before uvicorn serves, so "started" != "ready". Poll health so the
# script only returns once the endpoint can actually answer.
BASE="http://${LAYA_BIND:-127.0.0.1}:${LAYA_PORT:-8100}"
log "waiting for ${BASE}/health"
for i in $(seq 1 60); do
  if curl -fsS -m 5 "${BASE}/health" >/dev/null 2>&1; then
    dev="$(curl -fsS -m 5 "${BASE}/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("device","?"))' 2>/dev/null)"
    log "ready on ${dev} -- ${BASE}"
    log "next: scripts/smoke-test.sh"
    exit 0
  fi
  container_running || die "container exited during startup -- check: docker logs ${CONTAINER_NAME}"
  sleep 2
done

die "timed out waiting for health. Check: docker logs ${CONTAINER_NAME}"
