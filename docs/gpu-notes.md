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

## Measured

| | |
|---|---|
| Weights (typed-decisions) | 808 MB |
| Image build | ~2 min |
| Weight download | ~5 s |
| Model load | ~3.6 s |
| CPU-fallback latency | ~61 ms/question (3 questions, batched) |
| GPU latency | not yet measured — card was occupied |

Upstream claims ~33 ms/question on GPU; confirm once the card is free.
