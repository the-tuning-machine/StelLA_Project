"""Profile a dense-vs-LoRA linear layer and generate optional Mosaic reports."""

from __future__ import annotations

import argparse
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

STEPS = 5
BATCH_SIZE = 16
IN_FEATURES = 4096
OUT_FEATURES = 4096
RANK = 16

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


class LoRALinear(nn.Module):
    """Linear layer with a frozen base projection and trainable LoRA adapters."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(requires_grad=False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection and the LoRA update."""
        return self.base(inputs) + self.lora_b(self.lora_a(inputs))


def build_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic inputs and labels for the profiling run."""
    generator = torch.Generator(device="cpu").manual_seed(7)
    inputs = torch.randn(BATCH_SIZE, IN_FEATURES, generator=generator)
    labels = torch.randn(BATCH_SIZE, OUT_FEATURES, generator=generator)
    return inputs.to(DEVICE), labels.to(DEVICE)


def make_dense_model() -> nn.Module:
    """Construct the dense linear baseline model."""
    torch.manual_seed(7)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)


def make_lora_model() -> nn.Module:
    """Construct the frozen-base LoRA variant used for comparison."""
    torch.manual_seed(7)
    return LoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False), rank=RANK)


def profiler_activities() -> list[ProfilerActivity]:
    """Return the profiler activity set supported by the current device."""
    if DEVICE.type == "cuda":
        return [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    return [ProfilerActivity.CPU]


def bytes_to_gib(num_bytes: float) -> float:
    """Convert bytes to gibibytes."""
    return num_bytes / 1024**3


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
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.05
    )

    snapshot_path = SNAPSHOT_DIR / f"{name}_snapshot.pickle"
    trace_dir = SNAPSHOT_DIR / f"{name}_traces"

    with (
        capture_snapshot(snapshot_path),
        profile(
            activities=profiler_activities(),
            schedule=schedule(wait=0, warmup=0, active=STEPS, repeat=1),  # codespell:ignore warmup
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        ) as profiler,
    ):
        for _ in range(STEPS):
            profiler.step()

            with record_function("## forward ##"):
                pred = model(inputs)

            with record_function("## backward ##"):
                loss_fn(pred, labels).backward()

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    sys.stdout.write(f"{name}\n")
    sys.stdout.write(f"  snapshot: {snapshot_path}\n")
    sys.stdout.write(f"  trace_dir: {trace_dir}\n")


def load_mosaic() -> tuple[Any, Any, Any, Any]:
    """Import the Mosaic entry points required for report generation."""
    try:
        from mosaic.cmd.entry_point import (  # noqa: PLC0415
            get_memory_profile,
            get_memory_usage_by_annotation_stage,
            get_memory_usage_peak,
        )
        from mosaic.libmosaic.analyzer.memory_abstract import MemoryAbstract  # noqa: PLC0415
    except ImportError as exc:
        message = (
            "Mosaic is not importable in the current environment. Run `uv sync --group mosaic` "
            "to install the configured Mosaic dependency and its runtime requirements."
        )
        raise ImportError(message) from exc

    return (
        get_memory_profile,
        get_memory_usage_by_annotation_stage,
        get_memory_usage_peak,
        MemoryAbstract,
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


def analyze_snapshot(name: str) -> dict[str, Any]:
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

    return peak_summary


def write_comparison(summaries: list[dict[str, Any]]) -> Path:
    """Write a dense-vs-LoRA comparison JSON file."""
    summary_by_name = {summary["name"]: summary for summary in summaries}
    dense = summary_by_name["dense"]
    frozen_lora = summary_by_name["frozen_lora"]

    comparison = {
        "dense": dense,
        "frozen_lora": frozen_lora,
        "delta": {
            "dynamic_peak_bytes": frozen_lora["dynamic_peak_bytes"] - dense["dynamic_peak_bytes"],
            "dynamic_peak_gib": frozen_lora["dynamic_peak_gib"] - dense["dynamic_peak_gib"],
            "overall_peak_bytes": frozen_lora["overall_peak_bytes"] - dense["overall_peak_bytes"],
            "overall_peak_gib": frozen_lora["overall_peak_gib"] - dense["overall_peak_gib"],
        },
    }

    comparison_path = OUTPUT_DIR / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison_path


def run_mosaic_analysis() -> None:
    """Analyze all known snapshots and emit a comparison summary."""
    summaries = [analyze_snapshot(name) for name in SNAPSHOT_NAMES]
    comparison_path = write_comparison(summaries)
    sys.stdout.write(f"comparison: {comparison_path}\n")


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
