"""Measure per-question latency on GPU, to check the ~33 ms/question claim.

Reports median and p95 rather than a mean: CUDA timings are right-skewed, so a mean
flatters or panics depending on which outlier landed in the sample.
"""

import os
import statistics
import sys
import time

import torch

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/models/laya-typed-decisions")
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "10"))

STATE = (
    "Subject: URGENT - production API returning 500s\n\n"
    "Our checkout has been down for 20 minutes and we are losing orders. "
    "We are on the Enterprise plan. Please escalate immediately."
)

ONE = {
    "intent": {
        "type": "choice",
        "instructions": "What does the sender want?",
        "criteria": {
            "support": "needs help with a broken product",
            "sales": "wants to buy or upgrade",
            "billing": "has a question about an invoice or payment",
        },
    }
}
THREE = dict(ONE)
THREE["urgency"] = {
    "type": "score",
    "instructions": "How urgent is this message?",
    "criteria": ["not urgent", "soon", "critical"],
}
THREE["is_spam"] = {"type": "noul", "instructions": "Is this message spam?"}


def bench(agent, questions, label):
    for _ in range(WARMUP):
        agent.system_one(STATE, questions)
    torch.cuda.synchronize()

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        agent.system_one(STATE, questions)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    median = statistics.median(samples)
    p95 = samples[int(len(samples) * 0.95) - 1]
    n = len(questions)
    print(
        f"{label:<24} median {median:7.2f} ms  p95 {p95:7.2f} ms  "
        f"min {samples[0]:6.2f} ms  |  per-question {median / n:6.2f} ms"
    )
    return median / n


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("FAIL: no CUDA device.")

    import laya

    agent = laya.load(MODEL_DIR, device="cuda")
    if agent.device.type != "cuda":
        sys.exit("FAIL: model fell back to CPU; benchmark would be meaningless.")

    print(f"device: {agent.device} | {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__} | warmup {WARMUP} | iters {ITERS}\n")

    per_q_1 = bench(agent, ONE, "1 question")
    per_q_3 = bench(agent, THREE, "3 questions (batched)")

    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(
        "\nupstream claims ~33 ms/question; measured "
        f"{per_q_1:.1f} ms single, {per_q_3:.1f} ms batched-per-question."
    )


if __name__ == "__main__":
    main()
