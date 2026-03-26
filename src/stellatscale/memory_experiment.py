"""Reusable theory and comparison helpers for the dense-vs-LoRA memory experiment.

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

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def bytes_to_gib(num_bytes: float) -> float:
    """Convert bytes to gibibytes."""
    return num_bytes / 1024**3


class LinearModelVariant(StrEnum):
    """Supported linear-layer variants for the memory experiment."""

    DENSE = "dense"
    FROZEN_LORA = "frozen_lora"


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
        Rank of the LoRA adapters for the frozen-base variant.
    steps:
        Number of training steps to profile.
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
    def input_bytes(self) -> int:
        """Return the input tensor size in bytes."""
        return self.batch_size * self.in_features * self.activation_bytes

    @property
    def label_bytes(self) -> int:
        """Return the label tensor size in bytes."""
        return self.batch_size * self.out_features * self.activation_bytes


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


class FrozenLoRALinear(nn.Module):
    """Linear layer with a frozen base projection and trainable LoRA adapters."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        """Initialize the frozen base projection and low-rank adapters."""
        super().__init__()
        self.base = base
        self.base.requires_grad_(requires_grad=False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection and LoRA update."""
        return self.base(inputs) + self.lora_b(self.lora_a(inputs))


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
            "files": dict(self.files),
        }


@dataclass(frozen=True, slots=True)
class ComparisonTolerances:
    """Relative tolerances used when comparing theory and measurement."""

    static_baseline_relative: float = 0.05
    forward_dynamic_relative: float = 0.50
    backward_dynamic_relative: float = 0.50
    optimizer_dynamic_relative: float = 0.35


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
class TheoryExperimentComparison:
    """Comparison report preserving both agreement and disagreement."""

    name: str
    theory: TheoreticalMemorySummary
    measured: MemorySummary
    metrics: dict[str, ComparisonMetric]
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

    if variant is LinearModelVariant.DENSE:
        trainable_elements = dense_elements
        trainable_parameter_bytes = dense_parameter_bytes
        resident_parameter_bytes = dense_parameter_bytes
        forward_dynamic_estimate_bytes = (
            config.batch_size * (config.in_features + config.out_features) * config.activation_bytes
        )
        name = LinearModelVariant.DENSE.value
    else:
        trainable_elements = config.lora_trainable_parameter_count
        trainable_parameter_bytes = trainable_elements * config.parameter_bytes
        resident_parameter_bytes = dense_parameter_bytes + trainable_parameter_bytes
        forward_dynamic_estimate_bytes = (
            config.batch_size
            * (config.in_features + config.out_features + config.lora_rank)
            * config.activation_bytes
        )
        name = LinearModelVariant.FROZEN_LORA.value

    gradient_bytes = trainable_elements * config.gradient_bytes
    optimizer_state_bytes = trainable_elements * config.optimizer_state_bytes_per_trainable_element
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
    notes: tuple[str, ...] = (),
    possible_gap_sources: tuple[str, ...] = (),
) -> TheoryExperimentComparison:
    """Compare theoretical accounting against one measured summary.

    The report intentionally preserves disagreement so it can be investigated
    later rather than normalized away.
    """
    actual_tolerances = tolerances or ComparisonTolerances()
    metrics = {
        "static_baseline": _approximate_metric(
            metric_name="static_baseline",
            predicted_bytes=float(theory.static_baseline_bytes),
            measured_bytes=measured.static_memory_bytes,
            tolerance=actual_tolerances.static_baseline_relative,
        ),
        "forward_end_dynamic": _approximate_metric(
            metric_name="forward_end_dynamic",
            predicted_bytes=float(theory.forward_dynamic_estimate_bytes),
            measured_bytes=measured.annotation_bytes("## forward ##", "END"),
            tolerance=actual_tolerances.forward_dynamic_relative,
        ),
        "backward_end_dynamic": _approximate_metric(
            metric_name="backward_end_dynamic",
            predicted_bytes=float(theory.backward_dynamic_estimate_bytes),
            measured_bytes=measured.annotation_bytes("## backward ##", "END"),
            tolerance=actual_tolerances.backward_dynamic_relative,
        ),
        "optimizer_end_dynamic": _approximate_metric(
            metric_name="optimizer_end_dynamic",
            predicted_bytes=float(theory.optimizer_dynamic_estimate_bytes),
            measured_bytes=measured.annotation_bytes("## optimizer ##", "END"),
            tolerance=actual_tolerances.optimizer_dynamic_relative,
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
        notes=notes,
        possible_gap_sources=possible_gap_sources,
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
