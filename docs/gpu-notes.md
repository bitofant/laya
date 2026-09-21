# GPU notes (RTX 5090 / Blackwell)

Findings from bringing up the first working slice. Verified on the test machine
(RTX 5090 32 GB, driver 615.71.09, Docker 29.8.1).

## sm_120 needs a cu128+ torch wheel

Blackwell is `sm_120`. A default-index `pip install torch>=2.0.0` can resolve to a build
with no sm_120 kernels: it imports fine, reports the device correctly, and only dies at the
first forward pass with `no kernel image is available for execution on the device`.

The image pins both the version and the index:

```dockerfile
pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1+cu128"
```

Verified good: `torch 2.9.1+cu128`, CUDA 12.8, arch list
`sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`.

`laya==0.3.4` requires only `torch>=2.0.0`, which the pinned wheel already satisfies, so
installing laya afterwards does not pull a replacement build. Keep that install order.

## laya.load() falls back to CPU silently

`laya.load(..., device="cuda")` does **not** raise when CUDA is unusable. It prints a
warning and returns a CPU agent. Inference still returns correct answers, ~10x slower.

A smoke test that only checks "did we get answers back" therefore passes on CPU. Two
guards in `docker/smoke_test.py` exist for this and should not be removed:

1. A free-VRAM check (< 2.0 GB aborts) *before* load.
2. `agent.device.type != "cuda"` → hard fail *after* load.

## The SDK's Blackwell warning is often a red herring

That fallback warning suggests the cause is an unsupported CUDA architecture and
recommends a nightly torch build. On this machine the actual cause was **VRAM
exhaustion by another process** — the arch support was already correct.

Read the `Reason:` line in the warning, and the arch list / VRAM figures the smoke test
prints, before believing the Blackwell advice.

Observed: a vLLM engine holding 30.8 GB of 32 GB left ~16 MiB free. Laya needs roughly
0.8 GB for weights plus activations, so it fell back to CPU. Check with:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## Cold start is ~14.6 s and must be paid at startup, not by the first caller

The first forward pass pays CUDA context setup, cuBLAS handle creation and kernel
autotune. Measured on the RTX 5090:

| request | latency |
|---|---|
| 1st after load (no warmup) | **14,675 ms** |
| new question shape | 189 ms |
| steady state | ~6.5 ms |

`server.py` therefore runs a warmup pass over all three question types during lifespan
startup, before uvicorn serves. With it, the first real request is ~9 ms. Do not remove
it: without warmup the first caller sees a 14-second hang that reads as a broken service.

The container healthcheck uses `start_period: 90s` to cover this.

## Measured

| | |
|---|---|
| Weights (typed-decisions) | 808 MB |
| Image build | ~2 min (cold) |
| Weight download | ~5 s |
| Model load | ~3.6 s |
| Startup warmup | ~14.6 s |
| Peak VRAM | **2.44 GB** |
| Latency, 1 question | **6.22 ms** median (p95 6.24) |
| Latency, 3 questions batched | 6.80 ms median → **2.27 ms/question** |
| Latency via HTTP, 3 questions | ~9–18 ms (includes JSON + network) |

Upstream claims ~33 ms/question. **Measured 6.2 ms — about 5x faster**, presumably
because the 5090 is newer than whatever produced the published figure. Batching questions
into one request is a further ~3x win per question, since they share the forward pass.

Reproduce with `scripts/bench-latency.sh` (needs the card free).

## Coexisting with another GPU service

Laya needs ~3 GB free (2.44 GB peak plus headroom). On the test machine a vLLM container
runs at `--gpu-memory-utilization 0.982`, leaving ~1.37 GB — **not enough**, so the two
cannot currently run at once.

To run both, vLLM needs roughly `--gpu-memory-utilization 0.88` *and* a lowered
`--num-gpu-blocks-override` (currently 6400). The utilization flag alone will not do it:
the block override pins KV cache size independently, so lowering only the percentage does
not free the expected memory.

Note that vLLM container was created with `docker run` and no compose file, so changing
flags means recreating it with all its mounts, 84 env vars and tuned args reproduced by
hand. Stopping and restarting the existing container (`docker stop` / `docker start`)
preserves the config exactly and takes ~55 s to return to healthy — much safer.
