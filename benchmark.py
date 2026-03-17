"""
Benchmark LoRA vs StelLA vs Transformer on training cost metrics.

Measures per architecture:
  - Trainable / total parameters
  - Optimizer state memory (Adam: 2 floats per trainable param)
  - Peak GPU/RSS memory during a training loop
  - Wall time breakdown: forward, backward, optimizer step
  - GPU utilization timeline & memory usage plots
"""

import time
import tracemalloc
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import LoRATransformer, StelLAAdamW, StelLATransformer, Transformer

# ── Config ───────────────────────────────────────────────────────────────────

N_EMBD = 512
N_HEAD = 8
N_LAYER = 6
BLOCK_SIZE = 256
SEQ_LEN = 128
BATCH = 16
WARMUP_STEPS = 5
BENCH_STEPS = 50
RANKS = [4, 16, 32, 64]

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")


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
    # Per-step GPU timeline data
    gpu_mem_allocated_mb: list[float] = field(default_factory=list)
    gpu_mem_reserved_mb: list[float] = field(default_factory=list)
    gpu_util_timeline: list[dict] = field(default_factory=list)
    fwd_times_ms: list[float] = field(default_factory=list)
    bwd_times_ms: list[float] = field(default_factory=list)
    step_times_ms: list[float] = field(default_factory=list)

    @property
    def avg_total_ms(self):
        return self.avg_forward_ms + self.avg_backward_ms + self.avg_step_ms

    @property
    def throughput(self):
        """Samples per second."""
        return BATCH / (self.avg_total_ms / 1000)


def _sync():
    """Synchronize GPU if available (needed for accurate timing)."""
    if USE_GPU:
        torch.cuda.synchronize()


def bench_model(name, model, optimizer, rank=None):
    model = model.to(DEVICE)
    x = torch.randn(BATCH, SEQ_LEN, N_EMBD, device=DEVICE)
    criterion = nn.MSELoss()
    target = torch.randn(BATCH, SEQ_LEN, N_EMBD, device=DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt_state_bytes = trainable * 2 * 4

    # Warmup
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    _sync()

    # Benchmark
    fwd_times, bwd_times, step_times = [], [], []
    gpu_mem_alloc, gpu_mem_reserved = [], []
    gpu_util_timeline = []  # per-step breakdown

    if USE_GPU:
        torch.cuda.reset_peak_memory_stats()
    else:
        tracemalloc.start()
        tracemalloc.reset_peak()

    for step_i in range(BENCH_STEPS):
        optimizer.zero_grad()

        # -- Record GPU memory before forward --
        if USE_GPU:
            _sync()
            mem_before = torch.cuda.memory_allocated() / 1e6

        # Forward
        if USE_GPU:
            start_fwd = torch.cuda.Event(enable_timing=True)
            end_fwd = torch.cuda.Event(enable_timing=True)
            start_fwd.record()
        else:
            t0 = time.perf_counter()

        out = model(x)
        loss = criterion(out, target)

        if USE_GPU:
            end_fwd.record()
            _sync()
            fwd_ms = start_fwd.elapsed_time(end_fwd)
            mem_after_fwd = torch.cuda.memory_allocated() / 1e6
        else:
            t1 = time.perf_counter()
            fwd_ms = (t1 - t0) * 1000

        # Backward
        if USE_GPU:
            start_bwd = torch.cuda.Event(enable_timing=True)
            end_bwd = torch.cuda.Event(enable_timing=True)
            start_bwd.record()
        else:
            t1 = time.perf_counter()

        loss.backward()

        if USE_GPU:
            end_bwd.record()
            _sync()
            bwd_ms = start_bwd.elapsed_time(end_bwd)
            mem_after_bwd = torch.cuda.memory_allocated() / 1e6
        else:
            t2 = time.perf_counter()
            bwd_ms = (t2 - t1) * 1000

        # Optimizer step
        if USE_GPU:
            start_step = torch.cuda.Event(enable_timing=True)
            end_step = torch.cuda.Event(enable_timing=True)
            start_step.record()
        else:
            t2 = time.perf_counter()

        optimizer.step()

        if USE_GPU:
            end_step.record()
            _sync()
            step_ms = start_step.elapsed_time(end_step)
            mem_after_step = torch.cuda.memory_allocated() / 1e6
        else:
            t3 = time.perf_counter()
            step_ms = (t3 - t2) * 1000

        fwd_times.append(fwd_ms)
        bwd_times.append(bwd_ms)
        step_times.append(step_ms)

        if USE_GPU:
            gpu_mem_alloc.append(torch.cuda.memory_allocated() / 1e6)
            gpu_mem_reserved.append(torch.cuda.memory_reserved() / 1e6)
            gpu_util_timeline.append({
                "step": step_i,
                "mem_before_mb": mem_before,
                "mem_after_fwd_mb": mem_after_fwd,
                "mem_after_bwd_mb": mem_after_bwd,
                "mem_after_step_mb": mem_after_step,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "step_ms": step_ms,
            })

    if USE_GPU:
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return BenchResult(
        name=name,
        rank=rank,
        trainable_params=trainable,
        total_params=total,
        optimizer_state_bytes=opt_state_bytes,
        peak_memory_bytes=peak_mem,
        avg_forward_ms=sum(fwd_times) / len(fwd_times),
        avg_backward_ms=sum(bwd_times) / len(bwd_times),
        avg_step_ms=sum(step_times) / len(step_times),
        gpu_mem_allocated_mb=gpu_mem_alloc,
        gpu_mem_reserved_mb=gpu_mem_reserved,
        gpu_util_timeline=gpu_util_timeline,
        fwd_times_ms=fwd_times,
        bwd_times_ms=bwd_times,
        step_times_ms=step_times,
    )


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_time_breakdown(results: list[BenchResult]):
    """Stacked bar chart: forward / backward / optimizer step per model+rank."""
    labels = [f"{r.name}\nr={r.rank}" if r.rank else r.name for r in results]
    fwd = [r.avg_forward_ms for r in results]
    bwd = [r.avg_backward_ms for r in results]
    step = [r.avg_step_ms for r in results]

    x = range(len(results))
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.8), 5))
    ax.bar(x, fwd, label="Forward", color="#4C72B0")
    ax.bar(x, bwd, bottom=fwd, label="Backward", color="#DD8452")
    ax.bar(x, step, bottom=[f + b for f, b in zip(fwd, bwd)], label="Optim step", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-step time breakdown (forward / backward / optimizer)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("benchmark_time_breakdown.png", dpi=150)
    plt.close(fig)
    print("  -> Saved benchmark_time_breakdown.png")



def plot_gpu_memory_phases(results: list[BenchResult]):
    """Per-phase memory footprint: before fwd, after fwd, after bwd, after step."""
    if not USE_GPU:
        print("  -> Skipping GPU memory phases (no GPU)")
        return

    gpu_results = [r for r in results if r.gpu_util_timeline]
    if not gpu_results:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(gpu_results) * 1.2), 5))
    x = range(len(gpu_results))
    width = 0.2

    # Average across steps for each phase
    phases = {
        "Before fwd": ("mem_before_mb", "#4C72B0"),
        "After fwd": ("mem_after_fwd_mb", "#DD8452"),
        "After bwd": ("mem_after_bwd_mb", "#55A868"),
        "After optim": ("mem_after_step_mb", "#C44E52"),
    }

    for i, (phase_name, (key, color)) in enumerate(phases.items()):
        vals = []
        for r in gpu_results:
            avg = sum(s[key] for s in r.gpu_util_timeline) / len(r.gpu_util_timeline)
            vals.append(avg)
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=phase_name, color=color)

    labels = [f"{r.name}\nr={r.rank}" if r.rank else r.name for r in gpu_results]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("GPU Memory (MB)")
    ax.set_title("GPU Memory per Training Phase (avg over steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("benchmark_gpu_memory_phases.png", dpi=150)
    plt.close(fig)
    print("  -> Saved benchmark_gpu_memory_phases.png")



def plot_throughput_comparison(results: list[BenchResult]):
    """Bar chart comparing throughput (samples/s) across models."""
    labels = [f"{r.name}\nr={r.rank}" if r.rank else r.name for r in results]
    throughputs = [r.throughput for r in results]
    colors = {"Transformer": "#4C72B0", "LoRA": "#DD8452", "StelLA": "#55A868"}

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.8), 5))
    bars = ax.bar(range(len(results)), throughputs,
                  color=[colors.get(r.name, "#999999") for r in results])
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Samples / second")
    ax.set_title("Training Throughput Comparison")

    # Add value labels on bars
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig("benchmark_throughput.png", dpi=150)
    plt.close(fig)
    print("  -> Saved benchmark_throughput.png")


def plot_peak_memory(results: list[BenchResult]):
    """Bar chart of peak memory per model."""
    labels = [f"{r.name}\nr={r.rank}" if r.rank else r.name for r in results]
    peak_mb = [r.peak_memory_bytes / 1e6 for r in results]
    colors = {"Transformer": "#4C72B0", "LoRA": "#DD8452", "StelLA": "#55A868"}

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.8), 5))
    bars = ax.bar(range(len(results)), peak_mb,
                  color=[colors.get(r.name, "#999999") for r in results])
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(f"Peak {'GPU' if USE_GPU else 'RSS'} Memory")

    for bar, val in zip(bars, peak_mb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig("benchmark_peak_memory.png", dpi=150)
    plt.close(fig)
    print("  -> Saved benchmark_peak_memory.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name()})" if USE_GPU else ""))

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

    # ── Generate plots ───────────────────────────────────────────────────────

    print("\nGenerating plots...")
    plot_time_breakdown(results)
    plot_gpu_memory_phases(results)
    plot_throughput_comparison(results)
    plot_peak_memory(results)
    print("Done!")


if __name__ == "__main__":
    main()
