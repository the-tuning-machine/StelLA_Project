"""Reusable theory and comparison helpers for the linear memory experiment.

The goal of this module is to keep theoretical accounting, measured summaries,
and the gap between the two as first-class data. Approximate agreement is
valuable, but disagreement is equally valuable because it can reveal problems in
theory, implementation, or the surrounding runtime environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def bytes_to_gib(num_bytes: float) -> float:
    """Convert bytes to gibibytes."""
    return num_bytes / 1024**3


class LinearModelVariant(StrEnum):
    """Supported linear-layer variants for the memory experiment."""

    LINEAR = "linear"
    LINEAR_LORA = "linear_lora"
    LINEAR_STELLA = "linear_stella"


@dataclass(frozen=True, slots=True)
class MemoryExperimentConfig:
    """Configuration shared by theory and runtime measurement.

    Parameters
    ----------
    batch_size:
        Batch size used by the experiment.
    in_features:
        Input width of the linear layer.
    out_features:
        Output width of the linear layer.
    lora_rank:
        Rank of the low-rank adaptation used by the LoRA and StelLA variants.
    warmup_steps:
        Number of warmup steps to run before the active profiling window begins.
    steps:
        Number of active training steps to profile after warmup.
    learning_rate:
        Optimizer learning rate.
    parameter_bytes:
        Bytes per stored parameter element.
    activation_bytes:
        Bytes per activation element.
    gradient_bytes:
        Bytes per gradient element.
    optimizer_state_bytes_per_trainable_element:
        Total bytes used by optimizer state per trainable element.
    include_input_in_static_baseline:
        Whether to count the input tensor as part of the static baseline.
    include_label_in_static_baseline:
        Whether to count the label tensor as part of the static baseline.
    """

    batch_size: int
    in_features: int
    out_features: int
    lora_rank: int
    warmup_steps: int = 5
    steps: int = 5
    learning_rate: float = 0.05
    parameter_bytes: int = 4
    activation_bytes: int = 4
    gradient_bytes: int = 4
    optimizer_state_bytes_per_trainable_element: int = 8
    include_input_in_static_baseline: bool = True
    include_label_in_static_baseline: bool = True

    def __post_init__(self) -> None:
        """Validate that the experiment dimensions are well-formed."""
        for field_name in (
            "batch_size",
            "in_features",
            "out_features",
            "lora_rank",
            "steps",
            "parameter_bytes",
            "activation_bytes",
            "gradient_bytes",
            "optimizer_state_bytes_per_trainable_element",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                message = f"{field_name} must be strictly positive, got {value}"
                raise ValueError(message)

        if self.warmup_steps < 0:
            message = f"warmup_steps must be non-negative, got {self.warmup_steps}"
            raise ValueError(message)

        if self.learning_rate <= 0:
            message = f"learning_rate must be strictly positive, got {self.learning_rate}"
            raise ValueError(message)

    @property
    def dense_parameter_count(self) -> int:
        """Return the dense weight element count."""
        return self.in_features * self.out_features

    @property
    def lora_trainable_parameter_count(self) -> int:
        """Return the trainable LoRA element count."""
        return self.lora_rank * (self.in_features + self.out_features)

    @property
    def stella_trainable_parameter_count(self) -> int:
        """Return the trainable StelLA element count for U, S, and V^T."""
        return self.lora_trainable_parameter_count + (self.lora_rank**2)

    @property
    def stella_optimizer_state_parameter_count(self) -> int:
        """Return the trainable StelLA element count tracked by Adam-style state."""
        return self.stella_trainable_parameter_count

    @property
    def input_bytes(self) -> int:
        """Return the input tensor size in bytes."""
        return self.batch_size * self.in_features * self.activation_bytes

    @property
    def label_bytes(self) -> int:
        """Return the label tensor size in bytes."""
        return self.batch_size * self.out_features * self.activation_bytes

    @property
    def total_profile_steps(self) -> int:
        """Return the total number of profiled iterations including warmup."""
        return self.warmup_steps + self.steps


@dataclass(frozen=True, slots=True)
class TheoreticalMemorySummary:
    """Theoretical memory accounting for one model variant."""

    name: str
    variant: LinearModelVariant
    resident_parameter_bytes: int
    trainable_parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    static_baseline_bytes: int
    forward_dynamic_estimate_bytes: int
    backward_dynamic_estimate_bytes: int
    optimizer_dynamic_estimate_bytes: int
    dynamic_peak_lower_bound_bytes: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert the theoretical summary to a JSON-serializable mapping."""
        return {
            "name": self.name,
            "variant": self.variant.value,
            "resident_parameter_bytes": self.resident_parameter_bytes,
            "resident_parameter_gib": bytes_to_gib(float(self.resident_parameter_bytes)),
            "trainable_parameter_bytes": self.trainable_parameter_bytes,
            "trainable_parameter_gib": bytes_to_gib(float(self.trainable_parameter_bytes)),
            "gradient_bytes": self.gradient_bytes,
            "gradient_gib": bytes_to_gib(float(self.gradient_bytes)),
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "optimizer_state_gib": bytes_to_gib(float(self.optimizer_state_bytes)),
            "static_baseline_bytes": self.static_baseline_bytes,
            "static_baseline_gib": bytes_to_gib(float(self.static_baseline_bytes)),
            "forward_dynamic_estimate_bytes": self.forward_dynamic_estimate_bytes,
            "forward_dynamic_estimate_gib": bytes_to_gib(
                float(self.forward_dynamic_estimate_bytes)
            ),
            "backward_dynamic_estimate_bytes": self.backward_dynamic_estimate_bytes,
            "backward_dynamic_estimate_gib": bytes_to_gib(
                float(self.backward_dynamic_estimate_bytes)
            ),
            "optimizer_dynamic_estimate_bytes": self.optimizer_dynamic_estimate_bytes,
            "optimizer_dynamic_estimate_gib": bytes_to_gib(
                float(self.optimizer_dynamic_estimate_bytes)
            ),
            "dynamic_peak_lower_bound_bytes": self.dynamic_peak_lower_bound_bytes,
            "dynamic_peak_lower_bound_gib": bytes_to_gib(
                float(self.dynamic_peak_lower_bound_bytes)
            ),
        }


@dataclass(frozen=True, slots=True)
class AnnotationMetadata:
    """Metadata attached to one measured annotation event."""

    stage: str
    name: str
    device: int
    time_us: int


@dataclass(frozen=True, slots=True)
class AnnotationMeasurement:
    """Measured memory value for one annotation event."""

    annotation: AnnotationMetadata
    memory_bytes: float
    memory_gib: float


@dataclass(frozen=True, slots=True)
class AllocatorMemoryState:
    """Measured CUDA allocator state at a steady-state point in time."""

    allocated_bytes: float
    allocated_gib: float
    reserved_bytes: float
    reserved_gib: float
    reserved_cached_bytes: float
    reserved_cached_gib: float

    def to_dict(self) -> dict[str, float]:
        """Convert the allocator state to a JSON-serializable mapping."""
        return {
            "allocated_bytes": self.allocated_bytes,
            "allocated_gib": self.allocated_gib,
            "reserved_bytes": self.reserved_bytes,
            "reserved_gib": self.reserved_gib,
            "reserved_cached_bytes": self.reserved_cached_bytes,
            "reserved_cached_gib": self.reserved_cached_gib,
        }


@dataclass(frozen=True, slots=True)
class TensorCategorySummary:
    """Summary of one live CUDA tensor category."""

    bytes: float
    gib: float

    def to_dict(self) -> dict[str, float]:
        """Convert the tensor category summary to a JSON-serializable mapping."""
        return {"bytes": self.bytes, "gib": self.gib}


@dataclass(frozen=True, slots=True)
class LiveTensorDescriptor:
    """Compact description of a live CUDA tensor not matched to a known category."""

    bytes: float
    gib: float
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool

    def to_dict(self) -> dict[str, object]:
        """Convert the live tensor descriptor to a JSON-serializable mapping."""
        return {
            "bytes": self.bytes,
            "gib": self.gib,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True, slots=True)
class LiveTensorAccounting:
    """Break down live CUDA tensor storage at a steady-state checkpoint."""

    total_live_tensor_bytes: float
    total_live_tensor_gib: float
    categories: dict[str, TensorCategorySummary]
    top_other_tensors: tuple[LiveTensorDescriptor, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Convert the live tensor accounting to a JSON-serializable mapping."""
        return {
            "total_live_tensor_bytes": self.total_live_tensor_bytes,
            "total_live_tensor_gib": self.total_live_tensor_gib,
            "categories": {key: value.to_dict() for key, value in self.categories.items()},
            "top_other_tensors": [tensor.to_dict() for tensor in self.top_other_tensors],
        }


@dataclass(frozen=True, slots=True)
class MemorySummary:
    """Measured memory summary loaded from Mosaic output."""

    name: str
    dynamic_peak_bytes: float
    dynamic_peak_gib: float
    static_memory_bytes: float
    static_memory_gib: float
    overall_peak_bytes: float
    overall_peak_gib: float
    annotation_memory: dict[str, AnnotationMeasurement]
    allocator_state: AllocatorMemoryState | None = None
    live_tensor_accounting: LiveTensorAccounting | None = None
    files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> MemorySummary:
        """Build a typed memory summary from a JSON-like mapping."""
        annotation_payload = _require_mapping(payload.get("annotation_memory"), "annotation_memory")
        files_payload = payload.get("files")
        files: dict[str, str] = {}
        if files_payload is not None:
            files_mapping = _require_mapping(files_payload, "files")
            files = {
                key: _require_str(value, f"files[{key}]") for key, value in files_mapping.items()
            }
        allocator_state_payload = payload.get("allocator_state")
        allocator_state = None
        if allocator_state_payload is not None:
            allocator_state = _parse_allocator_memory_state(allocator_state_payload)
        live_tensor_accounting_payload = payload.get("live_tensor_accounting")
        live_tensor_accounting = None
        if live_tensor_accounting_payload is not None:
            live_tensor_accounting = _parse_live_tensor_accounting(live_tensor_accounting_payload)

        return cls(
            name=_require_str(payload.get("name"), "name"),
            dynamic_peak_bytes=_require_float(
                payload.get("dynamic_peak_bytes"), "dynamic_peak_bytes"
            ),
            dynamic_peak_gib=_require_float(payload.get("dynamic_peak_gib"), "dynamic_peak_gib"),
            static_memory_bytes=_require_float(
                payload.get("static_memory_bytes"), "static_memory_bytes"
            ),
            static_memory_gib=_require_float(payload.get("static_memory_gib"), "static_memory_gib"),
            overall_peak_bytes=_require_float(
                payload.get("overall_peak_bytes"), "overall_peak_bytes"
            ),
            overall_peak_gib=_require_float(payload.get("overall_peak_gib"), "overall_peak_gib"),
            annotation_memory={
                key: _parse_annotation_measurement(key, value)
                for key, value in annotation_payload.items()
            },
            allocator_state=allocator_state,
            live_tensor_accounting=live_tensor_accounting,
            files=files,
        )

    @classmethod
    def from_json_path(cls, path: Path) -> MemorySummary:
        """Load a measured memory summary from a JSON file."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(_require_mapping(payload, str(path)))

    def annotation_bytes(self, annotation_name: str, stage: str, occurrence: int = 0) -> float:
        """Return the measured bytes for a given annotation event."""
        suffix = "" if occurrence == 0 else f"({occurrence})"
        key = f"{annotation_name}_{stage}{suffix}"
        measurement = self.annotation_memory.get(key)
        if measurement is None:
            message = f"Annotation {key!r} was not found in summary {self.name!r}"
            raise KeyError(message)
        return measurement.memory_bytes

    def annotation_delta_bytes(
        self,
        start_annotation_name: str,
        start_stage: str,
        end_annotation_name: str,
        end_stage: str,
        occurrence: int = 0,
    ) -> float:
        """Return the measured byte delta between two annotation events."""
        start_bytes = self.annotation_bytes(start_annotation_name, start_stage, occurrence)
        end_bytes = self.annotation_bytes(end_annotation_name, end_stage, occurrence)
        return end_bytes - start_bytes

    def to_dict(self) -> dict[str, object]:
        """Convert the measured summary back to a JSON-serializable mapping."""
        return {
            "name": self.name,
            "dynamic_peak_bytes": self.dynamic_peak_bytes,
            "dynamic_peak_gib": self.dynamic_peak_gib,
            "static_memory_bytes": self.static_memory_bytes,
            "static_memory_gib": self.static_memory_gib,
            "overall_peak_bytes": self.overall_peak_bytes,
            "overall_peak_gib": self.overall_peak_gib,
            "annotation_memory": {
                key: {
                    "annotation": {
                        "stage": value.annotation.stage,
                        "name": value.annotation.name,
                        "device": value.annotation.device,
                        "time_us": value.annotation.time_us,
                    },
                    "memory_bytes": value.memory_bytes,
                    "memory_gib": value.memory_gib,
                }
                for key, value in self.annotation_memory.items()
            },
            "allocator_state": (
                None if self.allocator_state is None else self.allocator_state.to_dict()
            ),
            "live_tensor_accounting": (
                None
                if self.live_tensor_accounting is None
                else self.live_tensor_accounting.to_dict()
            ),
            "files": dict(self.files),
        }


@dataclass(frozen=True, slots=True)
class ComparisonTolerances:
    """Relative tolerances used when comparing theory and measurement."""

    static_baseline_relative: float = 0.05
    steady_state_floor_relative: float = 0.35
    backward_delta_relative: float = 0.50
    peak_over_floor_relative: float = 0.35


@dataclass(frozen=True, slots=True)
class ComparisonNarrative:
    """Optional notes that explain theory-vs-measurement gaps."""

    notes: tuple[str, ...] = ()
    possible_gap_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonMetric:
    """One theory-vs-measurement metric with explicit gap information."""

    metric_name: str
    comparison_kind: str
    predicted_bytes: float
    measured_bytes: float
    absolute_error_bytes: float
    relative_error: float
    tolerance: float | None
    within_tolerance: bool

    def to_dict(self) -> dict[str, float | str | bool | None]:
        """Convert the metric to a JSON-serializable mapping."""
        return {
            "metric_name": self.metric_name,
            "comparison_kind": self.comparison_kind,
            "predicted_bytes": self.predicted_bytes,
            "predicted_gib": bytes_to_gib(self.predicted_bytes),
            "measured_bytes": self.measured_bytes,
            "measured_gib": bytes_to_gib(self.measured_bytes),
            "absolute_error_bytes": self.absolute_error_bytes,
            "absolute_error_gib": bytes_to_gib(self.absolute_error_bytes),
            "relative_error": self.relative_error,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True, slots=True)
class MemoryAccountingBreakdown:
    """Break down theory, active tensors, reserved cache, and residual gap."""

    theoretical_model_memory_bytes: float
    measured_active_tensor_memory_bytes: float
    measured_reserved_cached_memory_bytes: float
    unexplained_gap_bytes: float
    measured_total_reserved_bytes: float
    measured_mosaic_static_memory_bytes: float
    tensor_vs_theory_gap_bytes: float

    def to_dict(self) -> dict[str, float]:
        """Convert the memory accounting breakdown to a JSON-serializable mapping."""
        return {
            "theoretical_model_memory_bytes": self.theoretical_model_memory_bytes,
            "theoretical_model_memory_gib": bytes_to_gib(self.theoretical_model_memory_bytes),
            "measured_active_tensor_memory_bytes": self.measured_active_tensor_memory_bytes,
            "measured_active_tensor_memory_gib": bytes_to_gib(
                self.measured_active_tensor_memory_bytes
            ),
            "measured_reserved_cached_memory_bytes": self.measured_reserved_cached_memory_bytes,
            "measured_reserved_cached_memory_gib": bytes_to_gib(
                self.measured_reserved_cached_memory_bytes
            ),
            "unexplained_gap_bytes": self.unexplained_gap_bytes,
            "unexplained_gap_gib": bytes_to_gib(self.unexplained_gap_bytes),
            "measured_total_reserved_bytes": self.measured_total_reserved_bytes,
            "measured_total_reserved_gib": bytes_to_gib(self.measured_total_reserved_bytes),
            "measured_mosaic_static_memory_bytes": self.measured_mosaic_static_memory_bytes,
            "measured_mosaic_static_memory_gib": bytes_to_gib(
                self.measured_mosaic_static_memory_bytes
            ),
            "tensor_vs_theory_gap_bytes": self.tensor_vs_theory_gap_bytes,
            "tensor_vs_theory_gap_gib": bytes_to_gib(self.tensor_vs_theory_gap_bytes),
        }


@dataclass(frozen=True, slots=True)
class TheoryExperimentComparison:
    """Comparison report preserving both agreement and disagreement."""

    name: str
    theory: TheoreticalMemorySummary
    measured: MemorySummary
    metrics: dict[str, ComparisonMetric]
    memory_accounting: MemoryAccountingBreakdown
    notes: tuple[str, ...] = ()
    possible_gap_sources: tuple[str, ...] = ()

    @property
    def failing_metrics(self) -> tuple[str, ...]:
        """Return the metric names that are outside tolerance."""
        return tuple(name for name, metric in self.metrics.items() if not metric.within_tolerance)

    def to_dict(self) -> dict[str, object]:
        """Convert the comparison report to a JSON-serializable mapping."""
        return {
            "name": self.name,
            "theory": self.theory.to_dict(),
            "measured": self.measured.to_dict(),
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
            "memory_accounting": self.memory_accounting.to_dict(),
            "failing_metrics": list(self.failing_metrics),
            "notes": list(self.notes),
            "possible_gap_sources": list(self.possible_gap_sources),
        }


def build_theoretical_summary(
    config: MemoryExperimentConfig, variant: LinearModelVariant
) -> TheoreticalMemorySummary:
    """Build the theoretical accounting for one model variant."""
    dense_elements = config.dense_parameter_count
    dense_parameter_bytes = dense_elements * config.parameter_bytes
    static_extras = 0
    if config.include_input_in_static_baseline:
        static_extras += config.input_bytes
    if config.include_label_in_static_baseline:
        static_extras += config.label_bytes

    if variant is LinearModelVariant.LINEAR:
        trainable_elements = dense_elements
        trainable_parameter_bytes = dense_parameter_bytes
        resident_parameter_bytes = dense_parameter_bytes
        forward_dynamic_estimate_bytes = (
            config.batch_size * (config.in_features + config.out_features) * config.activation_bytes
        )
        name = LinearModelVariant.LINEAR.value
        optimizer_state_elements = trainable_elements
    elif variant is LinearModelVariant.LINEAR_LORA:
        trainable_elements = config.lora_trainable_parameter_count
        trainable_parameter_bytes = trainable_elements * config.parameter_bytes
        resident_parameter_bytes = dense_parameter_bytes + trainable_parameter_bytes
        forward_dynamic_estimate_bytes = (
            config.batch_size
            * (config.in_features + config.out_features + config.lora_rank)
            * config.activation_bytes
        )
        name = LinearModelVariant.LINEAR_LORA.value
        optimizer_state_elements = trainable_elements
    else:
        trainable_elements = config.stella_trainable_parameter_count
        trainable_parameter_bytes = trainable_elements * config.parameter_bytes
        resident_parameter_bytes = dense_parameter_bytes + trainable_parameter_bytes
        forward_dynamic_estimate_bytes = (
            config.batch_size
            * (config.in_features + config.out_features + config.lora_rank)
            * config.activation_bytes
        )
        name = LinearModelVariant.LINEAR_STELLA.value
        optimizer_state_elements = config.stella_optimizer_state_parameter_count

    gradient_bytes = trainable_elements * config.gradient_bytes
    optimizer_state_bytes = (
        optimizer_state_elements * config.optimizer_state_bytes_per_trainable_element
    )
    static_baseline_bytes = resident_parameter_bytes + static_extras
    backward_dynamic_estimate_bytes = gradient_bytes + forward_dynamic_estimate_bytes
    optimizer_dynamic_estimate_bytes = optimizer_state_bytes + forward_dynamic_estimate_bytes
    dynamic_peak_lower_bound_bytes = max(
        forward_dynamic_estimate_bytes,
        backward_dynamic_estimate_bytes,
        optimizer_dynamic_estimate_bytes,
    )

    return TheoreticalMemorySummary(
        name=name,
        variant=variant,
        resident_parameter_bytes=resident_parameter_bytes,
        trainable_parameter_bytes=trainable_parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        static_baseline_bytes=static_baseline_bytes,
        forward_dynamic_estimate_bytes=forward_dynamic_estimate_bytes,
        backward_dynamic_estimate_bytes=backward_dynamic_estimate_bytes,
        optimizer_dynamic_estimate_bytes=optimizer_dynamic_estimate_bytes,
        dynamic_peak_lower_bound_bytes=dynamic_peak_lower_bound_bytes,
    )


def compare_theory_to_measurement(
    theory: TheoreticalMemorySummary,
    measured: MemorySummary,
    tolerances: ComparisonTolerances | None = None,
    active_occurrence: int = 0,
    narrative: ComparisonNarrative | None = None,
) -> TheoryExperimentComparison:
    """Compare theoretical accounting against one measured summary.

    Warmup iterations are intentionally excluded from the primary comparison.
    The main metrics use post-warmup steady-state memory so allocator warmup and
    one-time optimizer initialization are not conflated with per-step theory.
    """
    actual_tolerances = tolerances or ComparisonTolerances()
    actual_narrative = narrative or ComparisonNarrative()
    allocator_state = measured.allocator_state
    theoretical_model_memory_bytes = float(
        theory.static_baseline_bytes + theory.optimizer_state_bytes
    )
    live_tensor_accounting = measured.live_tensor_accounting
    measured_active_tensor_memory_bytes = (
        measured.static_memory_bytes
        if live_tensor_accounting is None
        else live_tensor_accounting.total_live_tensor_bytes
    )
    measured_reserved_cached_memory_bytes = (
        0.0 if allocator_state is None else allocator_state.reserved_cached_bytes
    )
    active_allocated_bytes = (
        measured_active_tensor_memory_bytes
        if allocator_state is None
        else allocator_state.allocated_bytes
    )
    measured_total_reserved_bytes = active_allocated_bytes + measured_reserved_cached_memory_bytes
    steady_state_floor_bytes = measured.annotation_bytes(
        "## forward ##", "START", occurrence=active_occurrence
    )
    backward_delta_bytes = measured.annotation_delta_bytes(
        "## forward ##", "END", "## backward ##", "END", occurrence=active_occurrence
    )
    peak_over_floor_bytes = measured.dynamic_peak_bytes - steady_state_floor_bytes
    memory_accounting = MemoryAccountingBreakdown(
        theoretical_model_memory_bytes=theoretical_model_memory_bytes,
        measured_active_tensor_memory_bytes=measured_active_tensor_memory_bytes,
        measured_reserved_cached_memory_bytes=measured_reserved_cached_memory_bytes,
        unexplained_gap_bytes=active_allocated_bytes - measured_active_tensor_memory_bytes,
        measured_total_reserved_bytes=measured_total_reserved_bytes,
        measured_mosaic_static_memory_bytes=measured.static_memory_bytes,
        tensor_vs_theory_gap_bytes=measured_active_tensor_memory_bytes
        - theoretical_model_memory_bytes,
    )
    metrics = {
        "static_baseline": _approximate_metric(
            metric_name="static_baseline",
            predicted_bytes=float(theory.static_baseline_bytes),
            measured_bytes=measured.static_memory_bytes,
            tolerance=actual_tolerances.static_baseline_relative,
        ),
        "steady_state_floor": _approximate_metric(
            metric_name="steady_state_floor",
            predicted_bytes=float(theory.optimizer_dynamic_estimate_bytes),
            measured_bytes=steady_state_floor_bytes,
            tolerance=actual_tolerances.steady_state_floor_relative,
        ),
        "backward_delta": _approximate_metric(
            metric_name="backward_delta",
            predicted_bytes=float(theory.gradient_bytes),
            measured_bytes=backward_delta_bytes,
            tolerance=actual_tolerances.backward_delta_relative,
        ),
        "peak_over_floor": _approximate_metric(
            metric_name="peak_over_floor",
            predicted_bytes=float(theory.optimizer_state_bytes),
            measured_bytes=peak_over_floor_bytes,
            tolerance=actual_tolerances.peak_over_floor_relative,
        ),
        "dynamic_peak_lower_bound": _lower_bound_metric(
            metric_name="dynamic_peak_lower_bound",
            predicted_bytes=float(theory.dynamic_peak_lower_bound_bytes),
            measured_bytes=measured.dynamic_peak_bytes,
        ),
    }
    return TheoryExperimentComparison(
        name=theory.name,
        theory=theory,
        measured=measured,
        metrics=metrics,
        memory_accounting=memory_accounting,
        notes=actual_narrative.notes,
        possible_gap_sources=actual_narrative.possible_gap_sources,
    )


def _approximate_metric(
    metric_name: str, predicted_bytes: float, measured_bytes: float, tolerance: float
) -> ComparisonMetric:
    """Build an approximate-equality metric."""
    absolute_error_bytes = measured_bytes - predicted_bytes
    relative_error = _relative_error(predicted_bytes, measured_bytes)
    return ComparisonMetric(
        metric_name=metric_name,
        comparison_kind="approximate",
        predicted_bytes=predicted_bytes,
        measured_bytes=measured_bytes,
        absolute_error_bytes=absolute_error_bytes,
        relative_error=relative_error,
        tolerance=tolerance,
        within_tolerance=relative_error <= tolerance,
    )


def _lower_bound_metric(
    metric_name: str, predicted_bytes: float, measured_bytes: float
) -> ComparisonMetric:
    """Build a lower-bound metric where measured values are expected to be at least the bound."""
    absolute_error_bytes = measured_bytes - predicted_bytes
    relative_error = _relative_error(predicted_bytes, measured_bytes)
    return ComparisonMetric(
        metric_name=metric_name,
        comparison_kind="lower_bound",
        predicted_bytes=predicted_bytes,
        measured_bytes=measured_bytes,
        absolute_error_bytes=absolute_error_bytes,
        relative_error=relative_error,
        tolerance=None,
        within_tolerance=measured_bytes >= predicted_bytes,
    )


def _relative_error(predicted_bytes: float, measured_bytes: float) -> float:
    """Return the relative error between predicted and measured bytes."""
    denominator = abs(predicted_bytes)
    if denominator == 0:
        return 0.0 if measured_bytes == 0 else float("inf")
    return abs(measured_bytes - predicted_bytes) / denominator


def _parse_annotation_measurement(key: str, value: object) -> AnnotationMeasurement:
    """Parse one annotation measurement mapping."""
    mapping = _require_mapping(value, key)
    annotation_mapping = _require_mapping(
        mapping.get("annotation"), f"annotation_memory[{key}].annotation"
    )
    annotation = AnnotationMetadata(
        stage=_require_str(annotation_mapping.get("stage"), f"annotation_memory[{key}].stage"),
        name=_require_str(annotation_mapping.get("name"), f"annotation_memory[{key}].name"),
        device=_require_int(annotation_mapping.get("device"), f"annotation_memory[{key}].device"),
        time_us=_require_int(
            annotation_mapping.get("time_us"), f"annotation_memory[{key}].time_us"
        ),
    )
    memory_bytes = _require_float(
        mapping.get("memory_bytes"), f"annotation_memory[{key}].memory_bytes"
    )
    memory_gib = _require_float(mapping.get("memory_gib"), f"annotation_memory[{key}].memory_gib")
    return AnnotationMeasurement(
        annotation=annotation, memory_bytes=memory_bytes, memory_gib=memory_gib
    )


def _parse_allocator_memory_state(value: object) -> AllocatorMemoryState:
    """Parse a measured allocator-state mapping."""
    mapping = _require_mapping(value, "allocator_state")
    allocated_bytes = _require_float(
        mapping.get("allocated_bytes"), "allocator_state.allocated_bytes"
    )
    reserved_bytes = _require_float(mapping.get("reserved_bytes"), "allocator_state.reserved_bytes")
    reserved_cached_bytes = _require_float(
        mapping.get("reserved_cached_bytes"), "allocator_state.reserved_cached_bytes"
    )
    return AllocatorMemoryState(
        allocated_bytes=allocated_bytes,
        allocated_gib=_require_float(mapping.get("allocated_gib"), "allocator_state.allocated_gib"),
        reserved_bytes=reserved_bytes,
        reserved_gib=_require_float(mapping.get("reserved_gib"), "allocator_state.reserved_gib"),
        reserved_cached_bytes=reserved_cached_bytes,
        reserved_cached_gib=_require_float(
            mapping.get("reserved_cached_gib"), "allocator_state.reserved_cached_gib"
        ),
    )


def _parse_live_tensor_accounting(value: object) -> LiveTensorAccounting:
    """Parse a live-tensor accounting mapping."""
    mapping = _require_mapping(value, "live_tensor_accounting")
    categories_mapping = _require_mapping(
        mapping.get("categories"), "live_tensor_accounting.categories"
    )
    top_other_payload = mapping.get("top_other_tensors")
    top_other_tensors: list[LiveTensorDescriptor] = []
    if top_other_payload is not None:
        if not isinstance(top_other_payload, list):
            message = "live_tensor_accounting.top_other_tensors must be a list"
            raise TypeError(message)
        top_other_tensors = [
            _parse_live_tensor_descriptor(item, index)
            for index, item in enumerate(top_other_payload)
        ]

    return LiveTensorAccounting(
        total_live_tensor_bytes=_require_float(
            mapping.get("total_live_tensor_bytes"), "live_tensor_accounting.total_live_tensor_bytes"
        ),
        total_live_tensor_gib=_require_float(
            mapping.get("total_live_tensor_gib"), "live_tensor_accounting.total_live_tensor_gib"
        ),
        categories={
            key: _parse_tensor_category_summary(category_value, key)
            for key, category_value in categories_mapping.items()
        },
        top_other_tensors=tuple(top_other_tensors),
    )


def _parse_tensor_category_summary(value: object, key: str) -> TensorCategorySummary:
    """Parse one live tensor category summary."""
    mapping = _require_mapping(value, f"live_tensor_accounting.categories[{key}]")
    return TensorCategorySummary(
        bytes=_require_float(
            mapping.get("bytes"), f"live_tensor_accounting.categories[{key}].bytes"
        ),
        gib=_require_float(mapping.get("gib"), f"live_tensor_accounting.categories[{key}].gib"),
    )


def _parse_live_tensor_descriptor(value: object, index: int) -> LiveTensorDescriptor:
    """Parse one unmatched live tensor descriptor."""
    mapping = _require_mapping(value, f"live_tensor_accounting.top_other_tensors[{index}]")
    shape_payload = mapping.get("shape")
    if not isinstance(shape_payload, list) or not all(
        isinstance(item, int) for item in shape_payload
    ):
        message = (
            f"live_tensor_accounting.top_other_tensors[{index}].shape must be a list of integers"
        )
        raise TypeError(message)
    return LiveTensorDescriptor(
        bytes=_require_float(
            mapping.get("bytes"), f"live_tensor_accounting.top_other_tensors[{index}].bytes"
        ),
        gib=_require_float(
            mapping.get("gib"), f"live_tensor_accounting.top_other_tensors[{index}].gib"
        ),
        shape=tuple(cast("list[int]", shape_payload)),
        dtype=_require_str(
            mapping.get("dtype"), f"live_tensor_accounting.top_other_tensors[{index}].dtype"
        ),
        requires_grad=_require_bool(
            mapping.get("requires_grad"),
            f"live_tensor_accounting.top_other_tensors[{index}].requires_grad",
        ),
    )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Validate that a value is a mapping."""
    if not isinstance(value, dict):
        message = f"{field_name} must be a mapping"
        raise TypeError(message)
    return cast("Mapping[str, object]", value)


def _require_str(value: object, field_name: str) -> str:
    """Validate that a value is a string."""
    if not isinstance(value, str):
        message = f"{field_name} must be a string"
        raise TypeError(message)
    return value


def _require_float(value: object, field_name: str) -> float:
    """Validate that a value is numeric and convert it to float."""
    if not isinstance(value, int | float):
        message = f"{field_name} must be numeric"
        raise TypeError(message)
    return float(value)


def _require_int(value: object, field_name: str) -> int:
    """Validate that a value is an integer."""
    if not isinstance(value, int):
        message = f"{field_name} must be an integer"
        raise TypeError(message)
    return value


def _require_bool(value: object, field_name: str) -> bool:
    """Validate that a value is a boolean."""
    if not isinstance(value, bool):
        message = f"{field_name} must be a boolean"
        raise TypeError(message)
    return value
