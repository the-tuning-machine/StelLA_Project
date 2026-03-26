"""Tests for reusable memory experiment helpers."""

from __future__ import annotations

import torch
from torch import nn

from stellatscale.memory_experiment import (
    ComparisonTolerances,
    FrozenLoRALinear,
    LinearModelVariant,
    MemoryExperimentConfig,
    MemorySummary,
    build_theoretical_summary,
    compare_theory_to_measurement,
)


def test_dense_theoretical_accounting_matches_exact_counts() -> None:
    """Dense accounting should match the closed-form formulas exactly."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )

    summary = build_theoretical_summary(config, LinearModelVariant.DENSE)

    assert summary.resident_parameter_bytes == 4096 * 4096 * 4
    assert summary.trainable_parameter_bytes == 4096 * 4096 * 4
    assert summary.gradient_bytes == 4096 * 4096 * 4
    assert summary.optimizer_state_bytes == 4096 * 4096 * 8
    assert summary.static_baseline_bytes == (4096 * 4096 * 4) + (16 * 4096 * 4) + (16 * 4096 * 4)


def test_frozen_lora_theoretical_accounting_matches_exact_counts() -> None:
    """Frozen-LoRA accounting should only charge gradients and optimizer state to adapters."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )

    summary = build_theoretical_summary(config, LinearModelVariant.FROZEN_LORA)

    adapter_bytes = 16 * (4096 + 4096) * 4
    assert summary.resident_parameter_bytes == (4096 * 4096 * 4) + adapter_bytes
    assert summary.trainable_parameter_bytes == adapter_bytes
    assert summary.gradient_bytes == adapter_bytes
    assert summary.optimizer_state_bytes == 16 * (4096 + 4096) * 8


def test_frozen_lora_linear_keeps_base_frozen_and_optimizer_state_scoped() -> None:
    """The frozen base weight should not receive gradients or optimizer state."""
    model = FrozenLoRALinear(nn.Linear(8, 8, bias=False), rank=2)
    inputs = torch.randn(4, 8)
    labels = torch.randn(4, 8)
    adam = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1
    )

    prediction = model(inputs)
    loss = nn.functional.mse_loss(prediction, labels)
    loss.backward()
    adam.step()

    assert model.base.weight.grad is None
    assert model.base.weight not in adam.state
    assert model.lora_a.weight in adam.state
    assert model.lora_b.weight in adam.state


def test_comparison_report_preserves_theoretical_gap_information() -> None:
    """Comparison reports should preserve large gaps instead of normalizing them away."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )
    theory = build_theoretical_summary(config, LinearModelVariant.FROZEN_LORA)
    measured = MemorySummary.from_mapping(
        {
            "name": "frozen_lora",
            "dynamic_peak_bytes": 2359296.0,
            "dynamic_peak_gib": 0.002197265625,
            "static_memory_bytes": 85196800.0,
            "static_memory_gib": 0.079345703125,
            "overall_peak_bytes": 87556096.0,
            "overall_peak_gib": 0.08154296875,
            "annotation_memory": {
                "## forward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## forward ##",
                        "device": 0,
                        "time_us": 1,
                    },
                    "memory_bytes": 263168.0,
                    "memory_gib": 0.00024509429931640625,
                },
                "## backward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## backward ##",
                        "device": 0,
                        "time_us": 2,
                    },
                    "memory_bytes": 786432.0,
                    "memory_gib": 0.000732421875,
                },
                "## optimizer ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## optimizer ##",
                        "device": 0,
                        "time_us": 3,
                    },
                    "memory_bytes": 1310720.0,
                    "memory_gib": 0.001220703125,
                },
            },
            "files": {},
        }
    )

    report = compare_theory_to_measurement(
        theory,
        measured,
        tolerances=ComparisonTolerances(static_baseline_relative=0.10),
        notes=("Preserve the error; it may reflect theory, implementation, or runtime behavior.",),
        possible_gap_sources=(
            "External GPU workload from scripts like scripts/gpu_keepalive_loop.sh can perturb allocator baselines if running concurrently.",
        ),
    )

    assert "static_baseline" in report.failing_metrics
    assert report.metrics["dynamic_peak_lower_bound"].within_tolerance
    assert "gpu_keepalive_loop.sh" in report.possible_gap_sources[0]


def test_dense_measurement_agreement_is_reasonable_for_major_metrics() -> None:
    """The dense measured baseline should be in rough agreement for the major metrics."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )
    theory = build_theoretical_summary(config, LinearModelVariant.DENSE)
    measured = MemorySummary.from_mapping(
        {
            "name": "dense",
            "dynamic_peak_bytes": 285736960.0,
            "dynamic_peak_gib": 0.26611328125,
            "static_memory_bytes": 67633152.0,
            "static_memory_gib": 0.06298828125,
            "overall_peak_bytes": 353370112.0,
            "overall_peak_gib": 0.3291015625,
            "annotation_memory": {
                "## forward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## forward ##",
                        "device": 0,
                        "time_us": 1,
                    },
                    "memory_bytes": 8781824.0,
                    "memory_gib": 0.0081787109375,
                },
                "## backward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## backward ##",
                        "device": 0,
                        "time_us": 2,
                    },
                    "memory_bytes": 84410368.0,
                    "memory_gib": 0.07861328125,
                },
                "## optimizer ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## optimizer ##",
                        "device": 0,
                        "time_us": 3,
                    },
                    "memory_bytes": 151519232.0,
                    "memory_gib": 0.14111328125,
                },
            },
            "files": {},
        }
    )

    report = compare_theory_to_measurement(theory, measured)

    assert report.metrics["static_baseline"].within_tolerance
    assert report.metrics["optimizer_end_dynamic"].within_tolerance
    assert report.metrics["dynamic_peak_lower_bound"].within_tolerance
