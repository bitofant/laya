#!/usr/bin/env bash
# Measure per-question GPU latency in the running container.
# Needs the GPU free -- see docs/gpu-notes.md.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_docker
container_running || die "container '${CONTAINER_NAME}' is not running -- run ./start.sh first"

log "benchmarking in ${CONTAINER_NAME}"
compose exec -T laya python /app/bench_latency.py
