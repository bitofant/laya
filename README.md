# laya

Dockerized setup for running and managing a self-hosted [Laya](https://huggingface.co/convaiinnovations/laya) instance on an NVIDIA GPU.

> **Status: early WIP.** The repo layout and script interface below are the target design. Not all scripts exist yet — see [Roadmap](#roadmap).

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
- `git` (this repo uses submodules)

Verify GPU passthrough before anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

## Quick start

```bash
git clone --recurse-submodules git@github.com:bitofant/laya.git
cd laya

./setup.sh    # build images, download model weights
./start.sh    # start (or resume) the laya container
./stop.sh     # stop it again
```

Already cloned without submodules? `git submodule update --init --recursive`

## Layout

```
.
├── AGENTS.md        # instructions for LLM agents working on this repo
├── setup.sh         # build containers, download model weights
├── start.sh         # start / resume the laya container
├── stop.sh          # stop the running container
├── scripts/         # internal helpers (called by the top-level scripts or by agents)
└── docs/            # persisted decisions & research notes
```

Conventions:

- **Top-level scripts are for humans.** `scripts/` is for everything called by other scripts or by agents.
- **Model weights are never committed.** They live in gitignored paths and are fetched by `setup.sh` (via a throwaway container).
- **No secrets in git.** `.env*`, `*.key`, `*.pem` and `secrets/` are gitignored.
- **Submodules over copy-paste** for upstream code.

## Roadmap

- [ ] `setup.sh` — image build + weight download via throwaway HF container
- [ ] `start.sh` / `stop.sh` — container lifecycle
- [ ] Upstream Laya SDK as a git submodule
- [ ] HTTP inference endpoint (typed questions in, calibrated answers out)
- [ ] Checkpoint selection / preloading (English, multilingual, typed-decisions)
- [ ] Health check + smoke test script
- [ ] Fine-tuning workflow

## Links

- Model card: https://huggingface.co/convaiinnovations/laya
- Typed-decisions checkpoint: https://huggingface.co/convaiinnovations/laya-typed-decisions
- Upstream SDK: https://github.com/NandhaKishorM/laya · [PyPI](https://pypi.org/project/laya/)
- Live demo: https://huggingface.co/spaces/convaiinnovations/laya-demo
- Project page: https://laya.convaiinnovations.com/

## License

The Laya model and upstream SDK are Apache-2.0. Licensing for this repo's own scripts is TBD.
