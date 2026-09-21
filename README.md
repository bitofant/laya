# laya

Dockerized setup for running and managing a self-hosted [Laya](https://huggingface.co/convaiinnovations/laya) instance on an NVIDIA GPU.

> **Status: working.** Build, weights, lifecycle and an HTTP inference endpoint all run on
> GPU today. Measured **6.2 ms/question** on an RTX 5090 — see [Performance](#performance).

## What is Laya?

Laya is an open-weights **System 1 decision model**: instead of generating text, it answers *typed questions* about a piece of state (an email, a ticket, a JSON blob) in a **single forward pass** and returns calibrated probability distributions.

| | |
|---|---|
| Architecture | Bidirectional transformer (ModernBERT-large / mmBERT-base) + decision head; every option scored at its own `[MASK]` token |
| Size | 421M params (English), 322M (multilingual) |
| Latency | ~33 ms per question on GPU |
| Context | 512 tokens (English), 1024 tokens (multilingual / typed-decisions) |
| Languages | English root checkpoint; multilingual checkpoint covers 100+ |
| License | Apache-2.0 (open weights, self-hostable) |

Question types: `choice` (pick one of N described options), `score` (ordinal scale), `noul` (yes/no).

```python
questions = {
    "intent": {
        "type": "choice",
        "instructions": "What does the sender want?",
        "criteria": {"support": "needs help", "sales": "wants to buy"},
    },
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["not urgent", "soon", "critical"]},
    "is_spam": {"type": "noul", "instructions": "Is this spam?"},
}
```

Good fit for email triage, guardrails, content routing and structured decision workflows — anything where you want a cheap, fast, calibrated answer instead of a chat completion.

⚠️ The *base* checkpoints are near chance on typed decisions zero-shot (~0.362 accuracy). Use the `typed-decisions` checkpoint or fine-tune for production accuracy.

## Why this repo?

Everything needed to stand up a Laya instance — build, weights, run, stop — behind a few scripts a human can actually remember. No hosted API, no per-token fee.

## Requirements

- Linux host with an NVIDIA GPU
  - Reference/test machine: single **RTX 5090** (Blackwell, 32 GB VRAM)
  - Weights are small (~808 MB English, ~647 MB multilingual), so far less VRAM is sufficient
- Recent NVIDIA driver
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`nvidia-docker`)
- `git`

Verify GPU passthrough before anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

## Quick start

```bash
git clone git@github.com:bitofant/laya.git
cd laya

./setup.sh              # check prereqs, build image, download weights (~2 min)
./start.sh              # start the container, wait until the endpoint is ready
scripts/smoke-test.sh   # verify GPU + a real typed-decision pass over HTTP
./stop.sh               # stop it again
```

The endpoint listens on `127.0.0.1:8100` (localhost only — see [Security](#security)).

```bash
curl -s localhost:8100/decide -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
  "state": "Our checkout has been down for 20 minutes and we are losing orders.",
  "questions": {
    "intent":  {"type": "choice", "instructions": "What does the sender want?",
                "criteria": {"support": "needs help", "sales": "wants to buy"}},
    "urgency": {"type": "score",  "instructions": "How urgent?",
                "criteria": ["not urgent", "soon", "critical"]},
    "is_spam": {"type": "noul",   "instructions": "Is this spam?"}
  }
}'
```

Returns each question's answer, full probability distribution, calibrated confidence and
`latency_ms`. The SDK's built-in question sets are available without writing any:

```bash
curl -s localhost:8100/presets     # triage, email, guard, moderation, router
curl -s localhost:8100/decide -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{"state": "Server is on fire!", "preset": "triage"}'
```

## API

| | | |
|---|---|---|
| `GET` | `/health` | Status, device, VRAM. `503` while loading. Never loads the model. |
| `GET` | `/presets` | Names of the SDK's built-in question sets. |
| `POST` | `/decide` | `{"state": ..., "questions": {...}}` or `{"state": ..., "preset": "triage"}` |

`state` accepts a string, a JSON object, or a list of conversation turns. Supply exactly
one of `questions` or `preset`. Malformed questions return `422`; an unknown preset `404`.

**Batch your questions.** They share one forward pass, so three questions in one request
cost ~2.3 ms each versus ~6.2 ms asked separately.

## Configuration

Environment variables, all optional:

| | | |
|---|---|---|
| `LAYA_PORT` | `8100` | Published port |
| `LAYA_BIND` | `127.0.0.1` | Bind address — see [Security](#security) |
| `LAYA_DEVICE` | `cuda` | Set `cpu` to serve on CPU deliberately |
| `LAYA_MODEL_REPO` | `convaiinnovations/laya-typed-decisions` | HF repo to fetch |
| `LAYA_MODEL_NAME` | `laya-typed-decisions` | Directory under `./models` |
| `LAYA_API_KEY` | — | Set a secret key to enable Bearer authentication |
| `HF_TOKEN` | — | Only needed for gated repos; never written to disk |

`setup.sh` is idempotent — re-run it freely. `--force` re-downloads weights.
`stop.sh --down` removes the container entirely (weights are kept).

By default this pulls the **typed-decisions** checkpoint (808 MB), not the base one — see
the warning above. Override with `LAYA_MODEL_REPO` / `LAYA_MODEL_NAME`.

## Layout

```
.
├── AGENTS.md           # instructions for LLM agents working on this repo
├── setup.sh            # check prereqs, build image, download weights
├── start.sh            # start / resume the laya container
├── stop.sh             # stop the running container
├── docker-compose.yml  # container lifecycle (driven by start.sh / stop.sh)
├── docker/             # Dockerfile + in-container smoke test
├── scripts/            # internal helpers (called by the top-level scripts or by agents)
├── models/             # downloaded weights (gitignored)
└── docs/               # persisted decisions & research notes
```

Conventions:

- **Top-level scripts are for humans.** `scripts/` is for everything called by other scripts or by agents.
- **Model weights are never committed.** They live in gitignored paths and are fetched by `setup.sh` (via a throwaway container).
- **No secrets in git.** `.env*`, `*.key`, `*.pem` and `secrets/` are gitignored.
- **The SDK is a pinned pip dependency** (`laya==0.3.4`), not vendored source.

## Performance

Measured on an RTX 5090 with the typed-decisions checkpoint (`scripts/bench-latency.sh`):

| | |
|---|---|
| 1 question | **6.22 ms** median (p95 6.24) |
| 3 questions, batched | 6.80 ms → **2.27 ms/question** |
| Peak VRAM | 2.44 GB |
| Startup (load + warmup) | ~18 s |

Upstream claims ~33 ms/question; this is roughly **5x faster** on a 5090.

The first forward pass costs ~14.6 s in CUDA context setup and kernel autotune. The
server pays that during startup, so callers never see it — the first real request is ~9 ms.

## Security

The endpoint supports **Bearer Token authentication**. If `LAYA_API_KEY` is set, requests to `/decide` and `/presets` must include the `Authorization: Bearer <key>` header.

It binds to `127.0.0.1` by default for that reason. Before setting `LAYA_BIND=0.0.0.0`, ensure you have a strong API key or a reverse proxy with auth in front of it.

## Troubleshooting

**`laya.load()` never fails loudly.** If CUDA is unusable it prints a warning and quietly
runs on CPU — correct answers, ~10x slower. `scripts/smoke-test.sh` hard-fails in that
case instead of reporting a false pass.

**The SDK's "Blackwell / RTX 50-series" warning is often wrong.** It blames your PyTorch
build, but the usual real cause is another process holding the VRAM:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Laya needs ~2 GB free. Details and measured timings in [`docs/gpu-notes.md`](docs/gpu-notes.md).

## Roadmap

- [x] `setup.sh` — image build + weight download via throwaway HF container
- [x] `start.sh` / `stop.sh` — container lifecycle
- [x] Smoke test script (GPU probe + real typed-decision pass)
- [x] HTTP inference endpoint (typed questions in, calibrated answers out)
- [x] Health check endpoint + latency benchmark
- [x] Authentication in front of the endpoint
- [ ] Checkpoint selection / preloading (English, multilingual, typed-decisions)
- [ ] Fine-tuning workflow

## Links

- Model card: https://huggingface.co/convaiinnovations/laya
- Typed-decisions checkpoint: https://huggingface.co/convaiinnovations/laya-typed-decisions
- Upstream SDK: https://github.com/NandhaKishorM/laya · [PyPI](https://pypi.org/project/laya/)
- Live demo: https://huggingface.co/spaces/convaiinnovations/laya-demo
- Project page: https://laya.convaiinnovations.com/

## License

The Laya model and upstream SDK are Apache-2.0. Licensing for this repo's own scripts is TBD.
