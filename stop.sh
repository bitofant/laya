#!/usr/bin/env bash
# Stop the running laya container. State is kept, so ./start.sh resumes it.
# Pass --down to remove the container entirely.

source "$(dirname "${BASH_SOURCE[0]}")/scripts/lib.sh"

require_docker

if [[ "${1:-}" == "--down" ]]; then
  log "removing ${CONTAINER_NAME}"
  compose down
  log "removed (weights in ${MODELS_DIR} are untouched)"
  exit 0
fi

if ! container_running; then
  log "not running: ${CONTAINER_NAME}"
  exit 0
fi

log "stopping ${CONTAINER_NAME}"
compose stop
log "stopped"
