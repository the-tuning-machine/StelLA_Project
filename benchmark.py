"""
Benchmark LoRA vs StelLA vs Transformer on training cost metrics.

Measures per architecture:
  - Trainable / total parameters
  - Optimizer state memory (Adam: 2 floats per trainable param)
  - Peak RSS memory during a training loop
  - Wall time breakdown: forward, backward, optimizer step
"""

import time
import tracemalloc
from dataclasses import dataclass

import torch
import torch.nn as nn

from models import LoRATransformer, StelLAAdamW, StelLATransformer, Transformer

# ── Config ───────────────────────────────────────────────────────────────────

N_EMBD = 8
N_HEAD = 2
N_LAYER = 1
BLOCK_SIZE = 16
SEQ_LEN = 5
BATCH = 32
WARMUP_STEPS = 5
BENCH_STEPS = 50
RANKS = [1, 2, 4, 8]


@dataclass
class BenchResult:
    name: str
    rank: int | None
    trainable_params: int
    total_params: int
    optimizer_state_bytes: int
    peak_memory_bytes: int
    avg_forward_ms: float
    avg_backward_ms: float
    avg_step_ms: float

    @property
    def avg_total_ms(self):
        return self.avg_forward_ms + self.avg_backward_ms + self.avg_step_ms

    @property
    def throughput(self):
        """Samples per second."""
        return BATCH / (self.avg_total_ms / 1000)


def bench_model(name, model, optimizer, rank=None):
    x = torch.randn(BATCH, SEQ_LEN, N_EMBD)
    criterion = nn.MSELoss()
    target = torch.randn(BATCH, SEQ_LEN, N_EMBD)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # Adam stores (exp_avg, exp_avg_sq) per trainable param, each is float32
    opt_state_bytes = trainable * 2 * 4

    # Warmup
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    # Benchmark
    fwd_times, bwd_times, step_times = [], [], []

    tracemalloc.start()
    tracemalloc.reset_peak()

    for _ in range(BENCH_STEPS):
        optimizer.zero_grad()

        t0 = time.perf_counter()
        out = model(x)
        loss = criterion(out, target)
        t1 = time.perf_counter()

        loss.backward()
        t2 = time.perf_counter()

        optimizer.step()
        t3 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        step_times.append(t3 - t2)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchResult(
        name=name,
        rank=rank,
        trainable_params=trainable,
        total_params=total,
        optimizer_state_bytes=opt_state_bytes,
        peak_memory_bytes=peak_mem,
        avg_forward_ms=sum(fwd_times) / len(fwd_times) * 1000,
        avg_backward_ms=sum(bwd_times) / len(bwd_times) * 1000,
        avg_step_ms=sum(step_times) / len(step_times) * 1000,
    )


def main():
    results: list[BenchResult] = []
    base_kw = {"n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "block_size": BLOCK_SIZE}

    for rank in RANKS:
        # Transformer (baseline, same for all ranks)
        m = Transformer(**base_kw)
        opt = torch.optim.AdamW(m.parameters(), lr=0.01)
        results.append(bench_model("Transformer", m, opt, rank=None))

        # LoRA
        m = LoRATransformer(**base_kw, rank=rank)
        opt = torch.optim.AdamW(m.parameters(), lr=0.01)
        results.append(bench_model("LoRA", m, opt, rank=rank))

        # StelLA
        m = StelLATransformer(**base_kw, rank=rank)
        opt = StelLAAdamW(m.parameters(), lr=0.01)
        results.append(bench_model("StelLA", m, opt, rank=rank))

    # ── Print results ────────────────────────────────────────────────────────

    header = (
        f"{'Model':<14} {'Rank':>4}  {'Train':>7} {'Total':>7}  "
        f"{'Opt MB':>6}  {'Peak MB':>7}  "
        f"{'Fwd ms':>7} {'Bwd ms':>7} {'Step ms':>7} {'Total ms':>8}  "
        f"{'Samp/s':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        rank_str = str(r.rank) if r.rank is not None else "-"
        print(
            f"{r.name:<14} {rank_str:>4}  {r.trainable_params:>7} {r.total_params:>7}  "
            f"{r.optimizer_state_bytes / 1e6:>6.2f}  {r.peak_memory_bytes / 1e6:>7.2f}  "
            f"{r.avg_forward_ms:>7.2f} {r.avg_backward_ms:>7.2f} {r.avg_step_ms:>7.2f} {r.avg_total_ms:>8.2f}  "
            f"{r.throughput:>7.0f}"
        )

    # ── StelLA overhead vs LoRA ──────────────────────────────────────────────

    print("\n── StelLA overhead vs LoRA (same rank) ──")
    for rank in RANKS:
        lora = next(r for r in results if r.name == "LoRA" and r.rank == rank)
        stella = next(r for r in results if r.name == "StelLA" and r.rank == rank)
        step_overhead = (stella.avg_step_ms - lora.avg_step_ms) / lora.avg_step_ms * 100
        total_overhead = (stella.avg_total_ms - lora.avg_total_ms) / lora.avg_total_ms * 100
        mem_overhead = (stella.peak_memory_bytes - lora.peak_memory_bytes) / lora.peak_memory_bytes * 100
        print(
            f"  rank={rank}: step +{step_overhead:+.1f}%, "
            f"total +{total_overhead:+.1f}%, "
            f"peak mem +{mem_overhead:+.1f}%"
        )


if __name__ == "__main__":
    main()
