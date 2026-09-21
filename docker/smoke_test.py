"""Smoke test: prove the GPU stack and a real typed-decision forward pass work.

Run inside the container via scripts/smoke-test.sh. Exits non-zero on any failure.
"""

import json
import os
import sys
import time

import torch

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/models/laya-typed-decisions")


def check_gpu() -> torch.device:
    if not torch.cuda.is_available():
        sys.exit("FAIL: torch.cuda.is_available() is False -- container has no GPU.")

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"GPU: {name} (sm_{major}{minor})")

    # The real sm_120 trap: capability reported fine, no kernels compiled for it.
    # A tiny matmul forces kernel launch and fails loudly here rather than mid-inference.
    arch_list = torch.cuda.get_arch_list()
    print(f"compiled archs: {', '.join(arch_list)}")
    try:
        probe = torch.randn(64, 64, device="cuda")
        torch.mm(probe, probe).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        sys.exit(
            f"FAIL: GPU kernel launch failed on sm_{major}{minor}: {exc}\n"
            f"      Image was built for: {arch_list}\n"
            f"      Rebuild with a cu128+ torch wheel (see docker/Dockerfile)."
        )
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_b / 1e9, total_b / 1e9
    print(f"VRAM: {free_gb:.1f} GB free of {total_gb:.1f} GB")
    # The checkpoint is ~0.8 GB plus activations. If another process (a vLLM engine, a
    # stray notebook) has eaten the card, laya.load() does NOT raise -- it prints a warning
    # and falls back to CPU, which is why this check exists before we get there.
    if free_gb < 2.0:
        hogs = os.popen(
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader 2>/dev/null"
        ).read().strip()
        sys.exit(
            f"FAIL: only {free_gb:.1f} GB VRAM free; laya would silently fall back to CPU.\n"
            + (f"      GPU is held by:\n        {hogs}\n" if hogs else "")
            + "      Free the card and re-run."
        )
    print("GPU kernel probe: ok")
    return torch.device("cuda")


def main() -> None:
    check_gpu()

    if not os.path.isdir(MODEL_DIR):
        sys.exit(f"FAIL: weights not found at {MODEL_DIR}. Run ./setup.sh first.")

    import laya

    print(f"laya {laya.__version__} | loading {MODEL_DIR}")
    t0 = time.perf_counter()
    agent = laya.load(MODEL_DIR, device="cuda")
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    # laya.load() degrades to CPU with only a printed warning, so a smoke test that
    # merely checks for answers will pass while running ~10x slow on the wrong device.
    if agent.device.type != "cuda":
        sys.exit(
            f"FAIL: model landed on {agent.device.type}, not cuda.\n"
            "      See the [laya] warning above for the reason (usually VRAM exhaustion).\n"
            "      Note: that warning blames the PyTorch/Blackwell build, which is often\n"
            "      wrong -- check the arch list and VRAM figures printed above first."
        )
    print(f"device: {agent.device}")

    state = (
        "Subject: URGENT - production API returning 500s\n\n"
        "Our checkout has been down for 20 minutes and we are losing orders. "
        "We are on the Enterprise plan. Please escalate immediately."
    )
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "What does the sender want?",
            "criteria": {
                "support": "needs help with a broken product",
                "sales": "wants to buy or upgrade",
                "billing": "has a question about an invoice or payment",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this message?",
            "criteria": ["not urgent", "soon", "critical"],
        },
        "is_spam": {"type": "noul", "instructions": "Is this message spam?"},
    }

    # Warm up once so the reported latency reflects steady state, not lazy CUDA init.
    agent.system_one(state, questions)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = agent.system_one(state, questions)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print("\n--- result ---")
    print(json.dumps(result, indent=2))
    print(f"\n{len(questions)} questions in {elapsed_ms:.1f} ms "
          f"({elapsed_ms / len(questions):.1f} ms/question)")

    answers = result.get("answers", {})
    missing = set(questions) - set(answers)
    if missing:
        sys.exit(f"FAIL: model returned no answer for: {sorted(missing)}")

    # Sanity, not accuracy: this checkpoint should not call an outage 'sales'.
    if answers["intent"]["choice"] != "support":
        print(f"WARN: expected intent=support, got {answers['intent']['choice']!r}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
