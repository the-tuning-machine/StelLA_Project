"""Profile a dense-vs-LoRA linear layer and generate optional Mosaic reports."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)

from stellatscale.memory_experiment import (
    ComparisonTolerances,
    FrozenLoRALinear,
    LinearModelVariant,
    MemoryExperimentConfig,
    MemorySummary,
    build_theoretical_summary,
    bytes_to_gib,
    compare_theory_to_measurement,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class MemorySnapshotProtocol(Protocol):
    """Minimal snapshot view needed for peak-memory summaries."""

    dynamic_memory_peak: float
    static_memory: float


class MemoryAbstractProtocol(Protocol):
    """Minimal Mosaic memory abstraction used by this script."""

    memory_snapshot: MemorySnapshotProtocol


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "deliverables" / "single_layer_lora_outputs"
OUTPUT_DIR = SNAPSHOT_DIR / "mosaic"
ANNOTATIONS = ("## forward ##", "## backward ##", "## optimizer ##")
SNAPSHOT_NAMES = ("dense", "frozen_lora")
COMPARISON_REPORT_PATH = OUTPUT_DIR / "theory_comparison.json"

EXPERIMENT_CONFIG = MemoryExperimentConfig(
    batch_size=16, in_features=4096, out_features=4096, lora_rank=16, steps=5, learning_rate=0.05
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for profiling and post-processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mosaic-only",
        action="store_true",
        help="Skip profiling and only run Mosaic analysis on existing snapshots.",
    )
    return parser.parse_args()


def build_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic inputs and labels for the profiling run."""
    generator = torch.Generator(device="cpu").manual_seed(7)
    inputs = torch.randn(
        EXPERIMENT_CONFIG.batch_size, EXPERIMENT_CONFIG.in_features, generator=generator
    )
    labels = torch.randn(
        EXPERIMENT_CONFIG.batch_size, EXPERIMENT_CONFIG.out_features, generator=generator
    )
    return inputs.to(DEVICE), labels.to(DEVICE)


def make_dense_model() -> nn.Module:
    """Construct the dense linear baseline model."""
    torch.manual_seed(7)
    return nn.Linear(EXPERIMENT_CONFIG.in_features, EXPERIMENT_CONFIG.out_features, bias=False)


def make_lora_model() -> nn.Module:
    """Construct the frozen-base LoRA variant used for comparison."""
    torch.manual_seed(7)
    return FrozenLoRALinear(
        nn.Linear(EXPERIMENT_CONFIG.in_features, EXPERIMENT_CONFIG.out_features, bias=False),
        rank=EXPERIMENT_CONFIG.lora_rank,
    )


def profiler_activities() -> list[ProfilerActivity]:
    """Return the profiler activity set supported by the current device."""
    if DEVICE.type == "cuda":
        return [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    return [ProfilerActivity.CPU]


@contextmanager
def capture_snapshot(snapshot_path: Path) -> Iterator[None]:
    """Record a CUDA allocator snapshot around a profiling region."""
    if DEVICE.type != "cuda":
        yield
        return

    torch.cuda.memory._record_memory_history(max_entries=100000)  # noqa: SLF001
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(str(snapshot_path))  # noqa: SLF001
        torch.cuda.memory._record_memory_history(enabled=None)  # noqa: SLF001


def run_profile(name: str, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> None:
    """Run the profiler and snapshot capture for one model variant."""
    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=EXPERIMENT_CONFIG.learning_rate,
    )

    snapshot_path = SNAPSHOT_DIR / f"{name}_snapshot.pickle"
    trace_dir = SNAPSHOT_DIR / f"{name}_traces"
    profile_schedule = schedule(wait=0, warmup=0, active=EXPERIMENT_CONFIG.steps, repeat=1)

    with (
        capture_snapshot(snapshot_path),
        profile(
            activities=profiler_activities(),
            schedule=profile_schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        ) as profiler,
    ):
        for _ in range(EXPERIMENT_CONFIG.steps):
            with record_function("## forward ##"):
                pred = model(inputs)

            with record_function("## backward ##"):
                loss_fn(pred, labels).backward()

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            profiler.step()

    sys.stdout.write(f"{name}\n")
    sys.stdout.write(f"  snapshot: {snapshot_path}\n")
    sys.stdout.write(f"  trace_dir: {trace_dir}\n")


def load_mosaic() -> tuple[Any, Any, Any, Any]:
    """Import the Mosaic entry points required for report generation."""
    try:
        entry_point = importlib.import_module("mosaic.cmd.entry_point")
        memory_abstract_module = importlib.import_module(
            "mosaic.libmosaic.analyzer.memory_abstract"
        )
    except ImportError as exc:
        message = (
            "Mosaic is not importable in the current environment. Run `uv sync --group mosaic` "
            "to install the configured Mosaic dependency and its runtime requirements."
        )
        raise ImportError(message) from exc

    return (
        entry_point.get_memory_profile,
        entry_point.get_memory_usage_by_annotation_stage,
        entry_point.get_memory_usage_peak,
        memory_abstract_module.MemoryAbstract,
    )


def capture_stdout(callback: Callable[[], Any]) -> tuple[Any, str]:
    """Run a callback and return both its result and captured standard output."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = callback()
    return result, buffer.getvalue()


def build_peak_summary(name: str, memory_abstract: MemoryAbstractProtocol) -> dict[str, Any]:
    """Build a JSON-serializable peak-memory summary for a snapshot."""
    dynamic_peak_bytes = float(memory_abstract.memory_snapshot.dynamic_memory_peak)
    static_memory_bytes = float(memory_abstract.memory_snapshot.static_memory)
    overall_peak_bytes = dynamic_peak_bytes + static_memory_bytes
    return {
        "name": name,
        "dynamic_peak_bytes": dynamic_peak_bytes,
        "dynamic_peak_gib": bytes_to_gib(dynamic_peak_bytes),
        "static_memory_bytes": static_memory_bytes,
        "static_memory_gib": bytes_to_gib(static_memory_bytes),
        "overall_peak_bytes": overall_peak_bytes,
        "overall_peak_gib": bytes_to_gib(overall_peak_bytes),
    }


def analyze_snapshot(name: str) -> MemorySummary:
    """Generate Mosaic reports and a summary JSON file for one snapshot."""
    snapshot_path = SNAPSHOT_DIR / f"{name}_snapshot.pickle"
    if not snapshot_path.exists():
        message = f"Snapshot not found: {snapshot_path}"
        raise FileNotFoundError(message)

    (
        get_memory_profile,
        get_memory_usage_by_annotation_stage,
        get_memory_usage_peak,
        _memory_abstract,
    ) = load_mosaic()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories_path = OUTPUT_DIR / f"{name}_categories.html"
    annotations_path = OUTPUT_DIR / f"{name}_annotations.html"
    peak_report_path = OUTPUT_DIR / f"{name}_peak.txt"
    summary_path = OUTPUT_DIR / f"{name}_summary.json"

    get_memory_profile(
        snapshot=str(snapshot_path),
        out_path=str(categories_path),
        profile="categories",
        sampling_rate=1,
        preserve_allocation_order=True,
    )
    get_memory_profile(
        snapshot=str(snapshot_path),
        out_path=str(annotations_path),
        profile="annotations",
        sampling_rate=1,
        preserve_allocation_order=True,
    )

    annotation_usage, annotation_output = capture_stdout(
        lambda: get_memory_usage_by_annotation_stage(
            snapshot=str(snapshot_path), annotation=ANNOTATIONS, paste=False
        )
    )
    peak_memory_abstract, peak_output = capture_stdout(
        lambda: get_memory_usage_peak(
            snapshot=str(snapshot_path),
            trace="",
            allocation="",
            action="alloc",
            paste=False,
            print_stack=True,
            upload_result=False,
        )
    )

    annotation_summary = {
        stage: {
            "annotation": metadata,
            "memory_bytes": float(memory_bytes),
            "memory_gib": bytes_to_gib(float(memory_bytes)),
        }
        for stage, (metadata, memory_bytes) in annotation_usage.items()
    }
    peak_summary = build_peak_summary(name, peak_memory_abstract)
    peak_summary["annotation_memory"] = annotation_summary
    peak_summary["files"] = {
        "snapshot": str(snapshot_path),
        "categories_html": str(categories_path),
        "annotations_html": str(annotations_path),
        "peak_report": str(peak_report_path),
    }

    peak_report_path.write_text(peak_output + "\n" + annotation_output, encoding="utf-8")
    summary_path.write_text(json.dumps(peak_summary, indent=2), encoding="utf-8")

    sys.stdout.write(f"{name}\n")
    sys.stdout.write(f"  categories_html: {categories_path}\n")
    sys.stdout.write(f"  annotations_html: {annotations_path}\n")
    sys.stdout.write(f"  peak_report: {peak_report_path}\n")
    sys.stdout.write(f"  summary: {summary_path}\n")

    return MemorySummary.from_mapping(peak_summary)


def write_comparison(summaries: list[MemorySummary]) -> Path:
    """Write a dense-vs-LoRA comparison JSON file."""
    summary_by_name = {summary.name: summary for summary in summaries}
    dense = summary_by_name["dense"]
    frozen_lora = summary_by_name["frozen_lora"]

    comparison = {
        "dense": dense.to_dict(),
        "frozen_lora": frozen_lora.to_dict(),
        "delta": {
            "dynamic_peak_bytes": frozen_lora.dynamic_peak_bytes - dense.dynamic_peak_bytes,
            "dynamic_peak_gib": frozen_lora.dynamic_peak_gib - dense.dynamic_peak_gib,
            "overall_peak_bytes": frozen_lora.overall_peak_bytes - dense.overall_peak_bytes,
            "overall_peak_gib": frozen_lora.overall_peak_gib - dense.overall_peak_gib,
        },
    }

    comparison_path = OUTPUT_DIR / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison_path


def write_theory_comparison(summaries: list[MemorySummary]) -> Path:
    """Write a theory-vs-experiment comparison report for each variant."""
    possible_gap_sources = [
        "Theoretical gaps may come from incomplete theory, implementation details, or runtime behavior.",
        "Allocator caching, autograd temporaries, and kernel workspaces can all move the measured result away from the simple tensor model.",
    ]
    keepalive_script = ROOT / "scripts" / "gpu_keepalive_loop.sh"
    if keepalive_script.exists():
        possible_gap_sources.append(
            "External GPU workload from scripts like scripts/gpu_keepalive_loop.sh can perturb allocator baselines if running concurrently."
        )

    comparisons = {}
    for summary in summaries:
        variant = LinearModelVariant(summary.name)
        theory = build_theoretical_summary(EXPERIMENT_CONFIG, variant)
        comparison = compare_theory_to_measurement(
            theory,
            summary,
            tolerances=ComparisonTolerances(),
            notes=(
                "Keep the theory-vs-experiment gap explicit: disagreement is a signal to investigate, not something to smooth away.",
            ),
            possible_gap_sources=tuple(possible_gap_sources),
        )
        comparisons[summary.name] = comparison.to_dict()

    COMPARISON_REPORT_PATH.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    return COMPARISON_REPORT_PATH


def run_mosaic_analysis() -> None:
    """Analyze all known snapshots and emit a comparison summary."""
    summaries = [analyze_snapshot(name) for name in SNAPSHOT_NAMES]
    comparison_path = write_comparison(summaries)
    theory_path = write_theory_comparison(summaries)
    sys.stdout.write(f"comparison: {comparison_path}\n")
    sys.stdout.write(f"theory_comparison: {theory_path}\n")


def main() -> None:
    """Run profiling and, when available, Mosaic post-processing."""
    args = parse_args()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mosaic_only:
        run_mosaic_analysis()
        return

    inputs, labels = build_inputs()

    run_profile("dense", make_dense_model(), inputs, labels)
    run_profile("frozen_lora", make_lora_model(), inputs, labels)

    if DEVICE.type == "cuda":
        run_mosaic_analysis()


if __name__ == "__main__":
    main()
