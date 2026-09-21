#!/usr/bin/env bash
# Shared config + helpers. Sourced by every other script; not meant to be run directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# Single source of truth for names/paths. docker-compose.yml reads these via .env-style
# defaults, so keep the values in sync if you change them there.
IMAGE_NAME="${LAYA_IMAGE:-laya:local}"
CONTAINER_NAME="${LAYA_CONTAINER:-laya}"
COMPOSE_PROJECT="${LAYA_PROJECT:-laya}"

# Default checkpoint. The base checkpoints score near chance (~0.362) on typed decisions
# zero-shot, so typed-decisions is the only sane default for a working smoke test.
MODEL_REPO="${LAYA_MODEL_REPO:-convaiinnovations/laya-typed-decisions}"
MODEL_NAME="${LAYA_MODEL_NAME:-laya-typed-decisions}"
MODELS_DIR="${REPO_ROOT}/models"
MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"

export IMAGE_NAME CONTAINER_NAME COMPOSE_PROJECT MODEL_REPO MODEL_NAME MODELS_DIR MODEL_DIR

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

compose() {
  docker compose --project-name "$COMPOSE_PROJECT" \
    --file "${REPO_ROOT}/docker-compose.yml" "$@"
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker not found in PATH"
  docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is it running? are you in the docker group?)"
  docker compose version >/dev/null 2>&1 || die "docker compose v2 plugin not found"
}

# Fail early and specifically: a missing container toolkit is the most common setup fault,
# and its native error ("could not select device driver") reads like a docker bug.
require_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- install the NVIDIA driver"
  docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi -L >/dev/null 2>&1 \
    || die "docker cannot see the GPU -- install/configure the NVIDIA Container Toolkit:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
}

weights_present() {
  # laya.load() needs all four of these; a partial download must not look like success.
  [[ -f "${MODEL_DIR}/rl_agent_config.json" \
  && -f "${MODEL_DIR}/model.safetensors" \
  && -d "${MODEL_DIR}/tokenizer" \
  && -d "${MODEL_DIR}/encoder" ]]
}

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == "true" ]]
}
