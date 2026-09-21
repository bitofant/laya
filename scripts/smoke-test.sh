#!/usr/bin/env bash
# Run one real typed-decision pass on the GPU inside the running container.
# Verifies: GPU visible, sm_120 kernels present, weights load, inference returns answers.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_docker

container_running || die "container '${CONTAINER_NAME}' is not running -- run ./start.sh first"

log "running smoke test in ${CONTAINER_NAME}"
compose exec -T laya python /app/smoke_test.py
