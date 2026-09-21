# Laya runtime image.
# No CUDA base image needed: the cu128 torch wheels bundle the CUDA runtime libs,
# and the NVIDIA Container Toolkit injects the driver at `docker run --gpus all`.
FROM python:3.12-slim

# RTX 5090 is Blackwell (sm_120). Only cu128+ builds ship sm_120 kernels; a default-index
# torch install can resolve to a build without them, which imports fine and then dies with
# "no kernel image is available for execution on the device" on the first forward pass.
# Pin the wheel AND the index. Do not relax to a bare `pip install torch`.
ARG TORCH_VERSION=2.9.1
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      "torch==${TORCH_VERSION}+cu128"

# laya depends on torch>=2.0.0, already satisfied above, so pip leaves the cu128 build alone.
ARG LAYA_VERSION=0.3.4
RUN pip install --no-cache-dir "laya==${LAYA_VERSION}"

# Weights are mounted at runtime (never baked into the image, never committed).
ENV LAYA_MODEL_DIR=/models/laya-typed-decisions \
    HF_HOME=/models/.hf \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY smoke_test.py /app/smoke_test.py

# Idle by default: this slice has no server yet, so start.sh keeps a warm container
# that scripts/smoke-test.sh can exec into. The HTTP endpoint replaces this CMD later.
CMD ["sleep", "infinity"]
