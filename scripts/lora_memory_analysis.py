"""Profile linear, linear-LoRA, and linear-StelLA variants and generate optional Mosaic reports."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import re
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

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
    ComparisonNarrative,
    ComparisonTolerances,
    LinearModelVariant,
    MemoryExperimentConfig,
    MemorySummary,
    build_theoretical_summary,
    bytes_to_gib,
    compare_theory_to_measurement,
)
from stellatscale.models import LoRALinear, StelLAAdamW, StelLALinear

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class MemorySnapshotProtocol(Protocol):
    """Minimal snapshot view needed for peak-memory summaries."""

    dynamic_memory_peak: float
    static_memory: float


class MemoryAbstractProtocol(Protocol):
    """Minimal Mosaic memory abstraction used by this script."""

    memory_snapshot: MemorySnapshotProtocol


def current_allocator_state() -> dict[str, float] | None:
    """Return the current CUDA allocator state, or None on non-CUDA runs."""
    if DEVICE.type != "cuda":
        return None

    allocated_bytes = float(torch.cuda.memory_allocated())
    reserved_bytes = float(torch.cuda.memory_reserved())
    reserved_cached_bytes = max(0.0, reserved_bytes - allocated_bytes)
    return {
        "allocated_bytes": allocated_bytes,
        "allocated_gib": bytes_to_gib(allocated_bytes),
        "reserved_bytes": reserved_bytes,
        "reserved_gib": bytes_to_gib(reserved_bytes),
        "reserved_cached_bytes": reserved_cached_bytes,
        "reserved_cached_gib": bytes_to_gib(reserved_cached_bytes),
    }


def current_live_tensor_accounting(
    model: nn.Module, optimizer: torch.optim.Optimizer, inputs: torch.Tensor, labels: torch.Tensor
) -> dict[str, Any] | None:
    """Return a categorized snapshot of live CUDA tensor storage."""
    if DEVICE.type != "cuda":
        return None

    categories = _empty_live_tensor_categories()
    top_other_tensors: list[dict[str, Any]] = []
    known_ptrs = _known_tensor_pointers(model, optimizer, inputs, labels)

    for data_ptr, tensor in _collect_live_cuda_tensors().items():
        storage_bytes = float(tensor.untyped_storage().nbytes())
        category = _live_tensor_category(data_ptr, known_ptrs)
        categories[category] += storage_bytes
        if category == "other":
            top_other_tensors.append(_describe_live_tensor(tensor, storage_bytes))

    top_other_tensors.sort(key=lambda item: float(item["bytes"]), reverse=True)
    total_live_tensor_bytes = float(sum(categories.values()))
    return {
        "total_live_tensor_bytes": total_live_tensor_bytes,
        "total_live_tensor_gib": bytes_to_gib(total_live_tensor_bytes),
        "categories": {
            key: {"bytes": value, "gib": bytes_to_gib(value)} for key, value in categories.items()
        },
        "top_other_tensors": top_other_tensors[:8],
    }


def _empty_live_tensor_categories() -> dict[str, float]:
    """Return zero-initialized live tensor categories."""
    return {
        "parameters": 0.0,
        "gradients": 0.0,
        "optimizer_state": 0.0,
        "inputs": 0.0,
        "labels": 0.0,
        "other": 0.0,
    }


def _known_tensor_pointers(
    model: nn.Module, optimizer: torch.optim.Optimizer, inputs: torch.Tensor, labels: torch.Tensor
) -> dict[str, set[int] | int]:
    """Collect data pointers for tensors with known semantic roles."""
    return {
        "parameters": {
            parameter.untyped_storage().data_ptr()
            for parameter in model.parameters()
            if parameter.is_cuda
        },
        "gradients": {
            parameter.grad.untyped_storage().data_ptr()
            for parameter in model.parameters()
            if parameter.grad is not None and parameter.grad.is_cuda
        },
        "optimizer_state": {
            value.untyped_storage().data_ptr()
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_cuda
        },
        "inputs": inputs.untyped_storage().data_ptr(),
        "labels": labels.untyped_storage().data_ptr(),
    }


def _collect_live_cuda_tensors() -> dict[int, torch.Tensor]:
    """Collect unique live CUDA tensors indexed by storage pointer."""
    seen_tensors: dict[int, torch.Tensor] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    data_ptr = obj.untyped_storage().data_ptr()
                    if data_ptr not in seen_tensors:
                        seen_tensors[data_ptr] = obj
            except (AttributeError, ReferenceError, RuntimeError):
                continue
    return seen_tensors


def _live_tensor_category(data_ptr: int, known_ptrs: dict[str, set[int] | int]) -> str:
    """Map a tensor storage pointer to its accounting category."""
    parameter_ptrs = cast("set[int]", known_ptrs["parameters"])
    gradient_ptrs = cast("set[int]", known_ptrs["gradients"])
    optimizer_state_ptrs = cast("set[int]", known_ptrs["optimizer_state"])
    input_ptr = cast("int", known_ptrs["inputs"])
    label_ptr = cast("int", known_ptrs["labels"])

    if data_ptr in parameter_ptrs:
        return "parameters"
    if data_ptr in gradient_ptrs:
        return "gradients"
    if data_ptr in optimizer_state_ptrs:
        return "optimizer_state"
    if data_ptr == input_ptr:
        return "inputs"
    if data_ptr == label_ptr:
        return "labels"
    return "other"


def _describe_live_tensor(tensor: torch.Tensor, storage_bytes: float) -> dict[str, Any]:
    """Build a compact description of one unmatched live CUDA tensor."""
    return {
        "bytes": storage_bytes,
        "gib": bytes_to_gib(storage_bytes),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "requires_grad": bool(tensor.requires_grad),
    }


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT_DIR = ROOT / "results" / "memory" / "single_layer_lora"
RESULTS_DIR = RESULTS_ROOT_DIR
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
TRACE_DIR = RESULTS_DIR / "traces"
OUTPUT_DIR = RESULTS_DIR / "mosaic"
ANNOTATIONS = ("## forward ##", "## backward ##", "## optimizer ##")
SNAPSHOT_NAMES = ("linear", "linear_lora", "linear_stella")
COMPARISON_REPORT_PATH = RESULTS_DIR / "theory_comparison.json"

DEFAULT_EXPERIMENT_CONFIG = MemoryExperimentConfig(
    batch_size=16,
    in_features=4096,
    out_features=4096,
    lora_rank=16,
    warmup_steps=5,
    steps=5,
    learning_rate=0.05,
)
EXPERIMENT_CONFIG = DEFAULT_EXPERIMENT_CONFIG

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_results_dir(config: MemoryExperimentConfig, output_tag: str | None) -> Path:
    """Return the results directory for one experiment configuration."""
    if output_tag is None and config == DEFAULT_EXPERIMENT_CONFIG:
        return RESULTS_ROOT_DIR

    tag = output_tag or (
        f"din_{config.in_features}_dout_{config.out_features}"
        f"_r_{config.lora_rank}_b_{config.batch_size}"
    )
    return RESULTS_ROOT_DIR / "runs" / tag


def configure_runtime_paths(results_dir: Path) -> None:
    """Update the module-level paths used by the profiling workflow."""
    global RESULTS_DIR, SNAPSHOT_DIR, TRACE_DIR, OUTPUT_DIR, COMPARISON_REPORT_PATH  # noqa: PLW0603
    RESULTS_DIR = results_dir
    SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
    TRACE_DIR = RESULTS_DIR / "traces"
    OUTPUT_DIR = RESULTS_DIR / "mosaic"
    COMPARISON_REPORT_PATH = RESULTS_DIR / "theory_comparison.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for profiling and post-processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=EXPERIMENT_CONFIG.batch_size)
    parser.add_argument("--in-features", type=int, default=EXPERIMENT_CONFIG.in_features)
    parser.add_argument("--out-features", type=int, default=EXPERIMENT_CONFIG.out_features)
    parser.add_argument("--lora-rank", type=int, default=EXPERIMENT_CONFIG.lora_rank)
    parser.add_argument("--warmup-steps", type=int, default=EXPERIMENT_CONFIG.warmup_steps)
    parser.add_argument("--steps", type=int, default=EXPERIMENT_CONFIG.steps)
    parser.add_argument("--learning-rate", type=float, default=EXPERIMENT_CONFIG.learning_rate)
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional output subdirectory name under results/memory/single_layer_lora/runs/.",
    )
    parser.add_argument(
        "--mosaic-only",
        action="store_true",
        help="Skip profiling and only run Mosaic analysis on existing snapshots.",
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> MemoryExperimentConfig:
    """Build the experiment configuration from parsed CLI arguments."""
    return MemoryExperimentConfig(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
        lora_rank=args.lora_rank,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )


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
    """Construct the plain linear baseline model."""
    torch.manual_seed(7)
    return nn.Linear(EXPERIMENT_CONFIG.in_features, EXPERIMENT_CONFIG.out_features, bias=False)


def make_lora_model() -> nn.Module:
    """Construct the linear-LoRA variant used for comparison."""
    torch.manual_seed(7)
    return LoRALinear(
        nn.Linear(EXPERIMENT_CONFIG.in_features, EXPERIMENT_CONFIG.out_features, bias=False),
        rank=EXPERIMENT_CONFIG.lora_rank,
    )


def make_stella_model() -> nn.Module:
    """Construct the linear-StelLA variant used for comparison."""
    torch.manual_seed(7)
    return StelLALinear(
        nn.Linear(EXPERIMENT_CONFIG.in_features, EXPERIMENT_CONFIG.out_features, bias=False),
        rank=EXPERIMENT_CONFIG.lora_rank,
    )


def make_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
    """Construct the optimizer used for one profiled model variant."""
    if name != LinearModelVariant.LINEAR_STELLA.value:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=EXPERIMENT_CONFIG.learning_rate,
        )

    if not isinstance(model, StelLALinear):
        message = f"Expected StelLALinear for {name}, got {type(model)!r}"
        raise TypeError(message)
    return StelLAAdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=EXPERIMENT_CONFIG.learning_rate,
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
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = make_optimizer(name, model)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SNAPSHOT_DIR / f"{name}_snapshot.pickle"
    trace_dir = TRACE_DIR / name
    runtime_state_path = RESULTS_DIR / f"{name}_allocator_state.json"
    live_tensor_path = RESULTS_DIR / f"{name}_live_tensors.json"
    profile_schedule = schedule(
        wait=0, warmup=EXPERIMENT_CONFIG.warmup_steps, active=EXPERIMENT_CONFIG.steps, repeat=1
    )

    steady_state_allocator = None
    steady_state_live_tensors = None
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
        for step_index in range(EXPERIMENT_CONFIG.total_profile_steps):
            if DEVICE.type == "cuda" and step_index == EXPERIMENT_CONFIG.warmup_steps:
                torch.cuda.synchronize()
                steady_state_allocator = current_allocator_state()
                steady_state_live_tensors = current_live_tensor_accounting(
                    model, optimizer, inputs, labels
                )

            with record_function("## forward ##"):
                pred = model(inputs)

            with record_function("## backward ##"):
                loss_fn(pred, labels).backward()

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            profiler.step()

    if steady_state_allocator is not None:
        runtime_state_path.write_text(
            json.dumps(steady_state_allocator, indent=2), encoding="utf-8"
        )
    if steady_state_live_tensors is not None:
        live_tensor_path.write_text(
            json.dumps(steady_state_live_tensors, indent=2), encoding="utf-8"
        )

    sys.stdout.write(f"{name}\n")
    sys.stdout.write(f"  snapshot: {snapshot_path}\n")
    sys.stdout.write(f"  trace_dir: {trace_dir}\n")
    if steady_state_allocator is not None:
        sys.stdout.write(f"  allocator_state: {runtime_state_path}\n")
    if steady_state_live_tensors is not None:
        sys.stdout.write(f"  live_tensors: {live_tensor_path}\n")


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


def _parse_mosaic_memory_size(memory_size: str) -> float:
    """Convert a Mosaic-formatted memory size string into bytes."""
    units = {
        "B": 1.0,
        "KB": 1024.0,
        "MB": 1024.0**2,
        "GB": 1024.0**3,
        "KIB": 1024.0,
        "MIB": 1024.0**2,
        "GIB": 1024.0**3,
    }
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]+)", memory_size.strip())
    if match is None:
        message = f"Unsupported Mosaic memory size: {memory_size!r}"
        raise ValueError(message)
    value = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = units.get(unit)
    if multiplier is None:
        message = f"Unsupported Mosaic memory unit: {unit!r}"
        raise ValueError(message)
    return value * multiplier


def _parse_mosaic_category_profile(output: str) -> tuple[float | None, dict[str, float]]:
    """Parse total allocated and category bytes from Mosaic's category-profile output."""
    total_allocated_bytes = None
    category_bytes: dict[str, float] = {}
    total_match = re.search(r"Total Allocated:\s+([0-9.]+[A-Za-z]+)", output)
    if total_match is not None:
        total_allocated_bytes = _parse_mosaic_memory_size(total_match.group(1))

    for match in re.finditer(r"AllocationType\.([A-Z_]+):\s+([0-9.]+[A-Za-z]+)", output):
        category_bytes[match.group(1)] = _parse_mosaic_memory_size(match.group(2))

    return total_allocated_bytes, category_bytes


def _build_mosaic_peak_breakdown(summary: dict[str, Any], categories_output: str) -> dict[str, Any]:
    """Build a runtime-oriented peak breakdown from Mosaic category output."""
    total_allocated_bytes, category_bytes = _parse_mosaic_category_profile(categories_output)
    static_memory_bytes = float(summary["static_memory_bytes"])
    total_peak_bytes = float(summary["overall_peak_bytes"])
    total_dynamic_bytes = (
        float(summary["dynamic_peak_bytes"])
        if total_allocated_bytes is None
        else total_allocated_bytes
    )
    categories = {
        "Static": static_memory_bytes,
        "Activation": category_bytes.get("ACTIVATION", 0.0),
        "Backward": category_bytes.get("BACKWARD", 0.0),
        "Optimizer": category_bytes.get("OPTIMIZER", 0.0),
        "Unknown": category_bytes.get("UNKNOWN", 0.0),
    }
    return {
        "total_peak_bytes": total_peak_bytes,
        "total_peak_gib": bytes_to_gib(total_peak_bytes),
        "total_dynamic_bytes": total_dynamic_bytes,
        "total_dynamic_gib": bytes_to_gib(total_dynamic_bytes),
        "categories": {
            key: {"bytes": value, "gib": bytes_to_gib(value)} for key, value in categories.items()
        },
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
    allocator_state_path = RESULTS_DIR / f"{name}_allocator_state.json"

    _categories_profile_result, categories_output = capture_stdout(
        lambda: get_memory_profile(
            snapshot=str(snapshot_path),
            out_path=str(categories_path),
            profile="categories",
            sampling_rate=1,
            preserve_allocation_order=True,
        )
    )
    _annotations_profile_result, annotations_profile_output = capture_stdout(
        lambda: get_memory_profile(
            snapshot=str(snapshot_path),
            out_path=str(annotations_path),
            profile="annotations",
            sampling_rate=1,
            preserve_allocation_order=True,
        )
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
    peak_summary["mosaic_peak_breakdown"] = _build_mosaic_peak_breakdown(
        peak_summary, categories_output
    )
    peak_summary["annotation_memory"] = annotation_summary
    if allocator_state_path.exists():
        peak_summary["allocator_state"] = json.loads(
            allocator_state_path.read_text(encoding="utf-8")
        )
    live_tensor_path = RESULTS_DIR / f"{name}_live_tensors.json"
    if live_tensor_path.exists():
        peak_summary["live_tensor_accounting"] = json.loads(
            live_tensor_path.read_text(encoding="utf-8")
        )
    peak_summary["files"] = {
        "snapshot": str(snapshot_path),
        "categories_html": str(categories_path),
        "annotations_html": str(annotations_path),
        "peak_report": str(peak_report_path),
        "allocator_state": str(allocator_state_path),
        "live_tensors": str(live_tensor_path),
    }

    peak_report_path.write_text(
        categories_output
        + "\n"
        + annotations_profile_output
        + "\n"
        + peak_output
        + "\n"
        + annotation_output,
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(peak_summary, indent=2), encoding="utf-8")

    sys.stdout.write(f"{name}\n")
    sys.stdout.write(f"  categories_html: {categories_path}\n")
    sys.stdout.write(f"  annotations_html: {annotations_path}\n")
    sys.stdout.write(f"  peak_report: {peak_report_path}\n")
    sys.stdout.write(f"  summary: {summary_path}\n")

    return MemorySummary.from_mapping(peak_summary)


def write_comparison(summaries: list[MemorySummary]) -> Path:
    """Write a comparison JSON file covering all profiled variants."""
    summary_by_name = {summary.name: summary for summary in summaries}
    linear = summary_by_name[LinearModelVariant.LINEAR.value]

    pairwise_deltas = {}
    for variant_name in (
        LinearModelVariant.LINEAR_LORA.value,
        LinearModelVariant.LINEAR_STELLA.value,
    ):
        variant_summary = summary_by_name[variant_name]
        pairwise_deltas[f"{variant_name}_minus_linear"] = {
            "dynamic_peak_bytes": variant_summary.dynamic_peak_bytes - linear.dynamic_peak_bytes,
            "dynamic_peak_gib": variant_summary.dynamic_peak_gib - linear.dynamic_peak_gib,
            "overall_peak_bytes": variant_summary.overall_peak_bytes - linear.overall_peak_bytes,
            "overall_peak_gib": variant_summary.overall_peak_gib - linear.overall_peak_gib,
        }

    linear_lora = summary_by_name[LinearModelVariant.LINEAR_LORA.value]
    linear_stella = summary_by_name[LinearModelVariant.LINEAR_STELLA.value]

    comparison = {
        LinearModelVariant.LINEAR.value: linear.to_dict(),
        LinearModelVariant.LINEAR_LORA.value: linear_lora.to_dict(),
        LinearModelVariant.LINEAR_STELLA.value: linear_stella.to_dict(),
        "delta": {
            **pairwise_deltas,
            "linear_stella_minus_linear_lora": {
                "dynamic_peak_bytes": linear_stella.dynamic_peak_bytes
                - linear_lora.dynamic_peak_bytes,
                "dynamic_peak_gib": linear_stella.dynamic_peak_gib - linear_lora.dynamic_peak_gib,
                "overall_peak_bytes": linear_stella.overall_peak_bytes
                - linear_lora.overall_peak_bytes,
                "overall_peak_gib": linear_stella.overall_peak_gib - linear_lora.overall_peak_gib,
            },
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = RESULTS_DIR / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison_path


def write_theory_comparison(summaries: list[MemorySummary]) -> Path:
    """Write a theory-vs-experiment comparison report for each variant."""
    possible_gap_sources = [
        "Theoretical gaps may come from incomplete theory, implementation details, or runtime behavior.",
        "Allocator caching, autograd temporaries, and kernel workspaces can all move the measured result away from the simple tensor model.",
    ]

    comparisons = {}
    for summary in summaries:
        variant = LinearModelVariant(summary.name)
        theory = build_theoretical_summary(EXPERIMENT_CONFIG, variant)
        variant_notes = [
            "Warmup iterations are excluded from the main theory comparison so the reported metrics reflect steady-state training rather than cold-start effects.",
            "The comparison now uses steady-state floor, backward delta, and peak-over-floor metrics instead of raw stage-end bytes because absolute stage-end values mix persistent allocator state with per-step transients.",
            "The memory_accounting block separates theoretical steady-state model memory, measured live tensor memory, measured reserved or cached allocator memory, and unexplained active allocations that are not visible as tracked live tensors.",
            "Keep the theory-vs-experiment gap explicit: disagreement is a signal to investigate, not something to smooth away.",
        ]
        if variant is LinearModelVariant.LINEAR_STELLA:
            variant_notes.append(
                "The current StelLA run uses the existing StelLAAdamW hook-based optimizer, so persistent optimizer state follows the full trainable StelLA factorization rather than a reduced custom state model."
            )
        comparison = compare_theory_to_measurement(
            theory,
            summary,
            tolerances=ComparisonTolerances(),
            active_occurrence=EXPERIMENT_CONFIG.warmup_steps,
            narrative=ComparisonNarrative(
                notes=tuple(variant_notes), possible_gap_sources=tuple(possible_gap_sources)
            ),
        )
        comparisons[summary.name] = comparison.to_dict()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_REPORT_PATH.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    return COMPARISON_REPORT_PATH


def _bytes_to_mib(num_bytes: float) -> float:
    """Convert bytes to mebibytes."""
    return num_bytes / 1024**2


def _measured_breakdown(summary: MemorySummary) -> dict[str, float]:
    """Build a widget-friendly measured breakdown from one experiment summary."""
    occurrence = EXPERIMENT_CONFIG.warmup_steps
    live_tensor_accounting = summary.live_tensor_accounting
    allocator_state = summary.allocator_state
    if live_tensor_accounting is None or allocator_state is None:
        message = (
            f"Measured breakdown requires allocator and live-tensor accounting for {summary.name}"
        )
        raise ValueError(message)

    parameters_bytes = float(live_tensor_accounting.categories["parameters"].bytes)
    optimizer_bytes = float(live_tensor_accounting.categories["optimizer_state"].bytes)
    activations_bytes = max(
        0.0,
        summary.annotation_delta_bytes(
            "## forward ##", "START", "## forward ##", "END", occurrence=occurrence
        ),
    )
    gradients_bytes = max(
        0.0,
        summary.annotation_delta_bytes(
            "## forward ##", "END", "## backward ##", "END", occurrence=occurrence
        ),
    )
    overhead_bytes = max(
        0.0,
        float(allocator_state.allocated_bytes)
        - float(live_tensor_accounting.total_live_tensor_bytes),
    )

    total_bytes = (
        parameters_bytes + optimizer_bytes + activations_bytes + gradients_bytes + overhead_bytes
    )
    return {
        "Overhead": _bytes_to_mib(overhead_bytes),
        "Parameters": _bytes_to_mib(parameters_bytes),
        "Activations": _bytes_to_mib(activations_bytes),
        "Gradients": _bytes_to_mib(gradients_bytes),
        "Optimizer": _bytes_to_mib(optimizer_bytes),
        "total_mib": _bytes_to_mib(total_bytes),
    }


def write_widget_breakdown(summaries: list[MemorySummary]) -> Path:
    """Write a compact measured breakdown for the plotting widget."""
    payload = {
        "config": {
            "batch_size": EXPERIMENT_CONFIG.batch_size,
            "in_features": EXPERIMENT_CONFIG.in_features,
            "out_features": EXPERIMENT_CONFIG.out_features,
            "lora_rank": EXPERIMENT_CONFIG.lora_rank,
            "warmup_steps": EXPERIMENT_CONFIG.warmup_steps,
            "steps": EXPERIMENT_CONFIG.steps,
        },
        "variants": {summary.name: _measured_breakdown(summary) for summary in summaries},
    }
    output_path = RESULTS_DIR / "widget_breakdown.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_mosaic_peak_breakdown(summaries: list[MemorySummary]) -> Path:
    """Write a Mosaic-aligned peak-runtime breakdown for plotting."""
    summary_payloads = {
        summary.name: json.loads(
            (OUTPUT_DIR / f"{summary.name}_summary.json").read_text(encoding="utf-8")
        )
        for summary in summaries
    }
    payload = {
        "config": {
            "batch_size": EXPERIMENT_CONFIG.batch_size,
            "in_features": EXPERIMENT_CONFIG.in_features,
            "out_features": EXPERIMENT_CONFIG.out_features,
            "lora_rank": EXPERIMENT_CONFIG.lora_rank,
            "warmup_steps": EXPERIMENT_CONFIG.warmup_steps,
            "steps": EXPERIMENT_CONFIG.steps,
        },
        "variants": {
            name: summary_payloads[name]["mosaic_peak_breakdown"] for name in SNAPSHOT_NAMES
        },
    }
    output_path = RESULTS_DIR / "mosaic_peak_breakdown.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_mosaic_analysis() -> None:
    """Analyze all known snapshots and emit a comparison summary."""
    summaries = [analyze_snapshot(name) for name in SNAPSHOT_NAMES]
    comparison_path = write_comparison(summaries)
    theory_path = write_theory_comparison(summaries)
    widget_breakdown_path = write_widget_breakdown(summaries)
    mosaic_peak_breakdown_path = write_mosaic_peak_breakdown(summaries)
    sys.stdout.write(f"comparison: {comparison_path}\n")
    sys.stdout.write(f"theory_comparison: {theory_path}\n")
    sys.stdout.write(f"widget_breakdown: {widget_breakdown_path}\n")
    sys.stdout.write(f"mosaic_peak_breakdown: {mosaic_peak_breakdown_path}\n")


def main() -> None:
    """Run profiling and, when available, Mosaic post-processing."""
    args = parse_args()
    config = build_experiment_config(args)

    global EXPERIMENT_CONFIG  # noqa: PLW0603
    EXPERIMENT_CONFIG = config
    configure_runtime_paths(build_results_dir(config, args.output_tag))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mosaic_only:
        run_mosaic_analysis()
        return

    inputs, labels = build_inputs()

    run_profile(LinearModelVariant.LINEAR.value, make_dense_model(), inputs, labels)
    run_profile(LinearModelVariant.LINEAR_LORA.value, make_lora_model(), inputs, labels)
    run_profile(LinearModelVariant.LINEAR_STELLA.value, make_stella_model(), inputs, labels)

    if DEVICE.type == "cuda":
        run_mosaic_analysis()


if __name__ == "__main__":
    main()
