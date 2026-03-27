"""Tests for reusable memory experiment helpers."""

from __future__ import annotations

import torch
from torch import nn

from stellatscale.memory_experiment import (
    ComparisonNarrative,
    ComparisonTolerances,
    LinearModelVariant,
    MemoryExperimentConfig,
    MemorySummary,
    build_theoretical_summary,
    compare_theory_to_measurement,
)
from stellatscale.models import LoRALinear, StelLAAdamW, StelLALinear


def test_dense_theoretical_accounting_matches_exact_counts() -> None:
    """Dense accounting should match the closed-form formulas exactly."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )

    summary = build_theoretical_summary(config, LinearModelVariant.LINEAR)

    assert summary.resident_parameter_bytes == 4096 * 4096 * 4
    assert summary.trainable_parameter_bytes == 4096 * 4096 * 4
    assert summary.gradient_bytes == 4096 * 4096 * 4
    assert summary.optimizer_state_bytes == 4096 * 4096 * 8
    assert summary.static_baseline_bytes == (4096 * 4096 * 4) + (16 * 4096 * 4) + (16 * 4096 * 4)


def test_lora_theoretical_accounting_matches_exact_counts() -> None:
    """LoRA accounting should only charge gradients and optimizer state to adapters."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )

    summary = build_theoretical_summary(config, LinearModelVariant.LINEAR_LORA)

    adapter_bytes = 16 * (4096 + 4096) * 4
    assert summary.resident_parameter_bytes == (4096 * 4096 * 4) + adapter_bytes
    assert summary.trainable_parameter_bytes == adapter_bytes
    assert summary.gradient_bytes == adapter_bytes
    assert summary.optimizer_state_bytes == 16 * (4096 + 4096) * 8


def test_stella_theoretical_accounting_matches_exact_counts() -> None:
    """StelLA accounting should charge full Adam state to all trainable PEFT factors."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )

    summary = build_theoretical_summary(config, LinearModelVariant.LINEAR_STELLA)

    stella_trainable_bytes = ((16 * (4096 + 4096)) + (16**2)) * 4
    assert summary.resident_parameter_bytes == (4096 * 4096 * 4) + stella_trainable_bytes
    assert summary.trainable_parameter_bytes == stella_trainable_bytes
    assert summary.gradient_bytes == stella_trainable_bytes
    assert summary.optimizer_state_bytes == ((16 * (4096 + 4096)) + (16**2)) * 8


def test_lora_linear_keeps_base_frozen_and_optimizer_state_scoped() -> None:
    """The base weight should stay frozen and only LoRA PEFT weights should get Adam state."""
    model = LoRALinear(nn.Linear(8, 8, bias=False), rank=2)
    inputs = torch.randn(4, 8)
    labels = torch.randn(4, 8)
    adam = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1
    )
    named_parameters = dict(model.named_parameters())

    prediction = model(inputs)
    loss = nn.functional.mse_loss(prediction, labels)
    loss.backward()
    adam.step()

    assert model.base_layer.weight.grad is None
    assert model.base_layer.weight not in adam.state
    assert named_parameters["model.base_model.model.linear.lora_A.default.weight"] in adam.state
    assert named_parameters["model.base_model.model.linear.lora_B.default.weight"] in adam.state


def test_stella_linear_keeps_base_frozen_and_optimizer_state_scoped() -> None:
    """The base weight should stay frozen.

    StelLAAdamW should track all trainable PEFT factors.
    """
    model = StelLALinear(nn.Linear(8, 8, bias=False), rank=2)
    inputs = torch.randn(4, 8)
    labels = torch.randn(4, 8)
    optimizer = StelLAAdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.1
    )
    named_parameters = dict(model.named_parameters())

    prediction = model(inputs)
    loss = nn.functional.mse_loss(prediction, labels)
    loss.backward()
    optimizer.step()

    assert model.base_layer.weight.grad is None
    assert model.base_layer.weight not in optimizer.state
    assert (
        named_parameters["model.base_model.model.linear.stella_U.default.weight"].grad is not None
    )
    assert (
        named_parameters["model.base_model.model.linear.stella_Vt.default.weight"].grad is not None
    )
    assert (
        named_parameters["model.base_model.model.linear.stella_U.default.weight"] in optimizer.state
    )
    assert (
        named_parameters["model.base_model.model.linear.stella_S.default.weight"] in optimizer.state
    )
    assert (
        named_parameters["model.base_model.model.linear.stella_Vt.default.weight"]
        in optimizer.state
    )


def test_comparison_report_preserves_theoretical_gap_information() -> None:
    """Comparison reports should preserve large gaps instead of normalizing them away."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )
    expected_linear_lora_steady_state_bytes = 69206016.0
    expected_reserved_cache_bytes = 17039360.0
    theory = build_theoretical_summary(config, LinearModelVariant.LINEAR_LORA)
    measured = MemorySummary.from_mapping(
        {
            "name": "linear_lora",
            "dynamic_peak_bytes": 2359296.0,
            "dynamic_peak_gib": 0.002197265625,
            "static_memory_bytes": 85196800.0,
            "static_memory_gib": 0.079345703125,
            "overall_peak_bytes": 87556096.0,
            "overall_peak_gib": 0.08154296875,
            "allocator_state": {
                "allocated_bytes": expected_linear_lora_steady_state_bytes,
                "allocated_gib": 0.064453125,
                "reserved_bytes": 86245376.0,
                "reserved_gib": 0.080322265625,
                "reserved_cached_bytes": expected_reserved_cache_bytes,
                "reserved_cached_gib": 0.015869140625,
            },
            "live_tensor_accounting": {
                "total_live_tensor_bytes": expected_linear_lora_steady_state_bytes,
                "total_live_tensor_gib": 0.064453125,
                "categories": {
                    "parameters": {"bytes": 67633152.0, "gib": 0.06298828125},
                    "gradients": {"bytes": 0.0, "gib": 0.0},
                    "optimizer_state": {"bytes": 1048576.0, "gib": 0.0009765625},
                    "inputs": {"bytes": 262144.0, "gib": 0.000244140625},
                    "labels": {"bytes": 262144.0, "gib": 0.000244140625},
                    "other": {"bytes": 0.0, "gib": 0.0},
                },
                "top_other_tensors": [],
            },
            "annotation_memory": {
                "## forward ##_START": {
                    "annotation": {
                        "stage": "START",
                        "name": "## forward ##",
                        "device": 0,
                        "time_us": 0,
                    },
                    "memory_bytes": 1310720.0,
                    "memory_gib": 0.001220703125,
                },
                "## forward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## forward ##",
                        "device": 0,
                        "time_us": 1,
                    },
                    "memory_bytes": 1311744.0,
                    "memory_gib": 0.0012216567993164062,
                },
                "## backward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## backward ##",
                        "device": 0,
                        "time_us": 2,
                    },
                    "memory_bytes": 1835008.0,
                    "memory_gib": 0.001708984375,
                },
            },
            "files": {},
        }
    )

    report = compare_theory_to_measurement(
        theory,
        measured,
        tolerances=ComparisonTolerances(static_baseline_relative=0.10),
        narrative=ComparisonNarrative(
            notes=(
                "Preserve the error; it may reflect theory, implementation, or runtime behavior.",
            ),
            possible_gap_sources=(
                "External GPU workload from scripts like scripts/gpu_keepalive_loop.sh can perturb allocator baselines if running concurrently.",
            ),
        ),
    )

    assert report.failing_metrics == ("static_baseline",)
    assert report.metrics["steady_state_floor"].within_tolerance
    assert report.metrics["backward_delta"].within_tolerance
    assert report.metrics["peak_over_floor"].within_tolerance
    assert report.metrics["dynamic_peak_lower_bound"].within_tolerance
    assert (
        report.memory_accounting.theoretical_model_memory_bytes
        == expected_linear_lora_steady_state_bytes
    )
    assert (
        report.memory_accounting.measured_active_tensor_memory_bytes
        == expected_linear_lora_steady_state_bytes
    )
    assert (
        report.memory_accounting.measured_reserved_cached_memory_bytes
        == expected_reserved_cache_bytes
    )
    assert report.memory_accounting.unexplained_gap_bytes == 0.0
    assert report.memory_accounting.tensor_vs_theory_gap_bytes == 0.0
    assert "gpu_keepalive_loop.sh" in report.possible_gap_sources[0]


def test_dense_measurement_agreement_is_reasonable_for_major_metrics() -> None:
    """The dense measured baseline should be in rough agreement for the major metrics."""
    config = MemoryExperimentConfig(
        batch_size=16, in_features=4096, out_features=4096, lora_rank=16
    )
    expected_linear_steady_state_bytes = 201850880.0
    expected_non_tensor_gap_bytes = 16777216.0
    theory = build_theoretical_summary(config, LinearModelVariant.LINEAR)
    measured = MemorySummary.from_mapping(
        {
            "name": "linear",
            "dynamic_peak_bytes": 285736960.0,
            "dynamic_peak_gib": 0.26611328125,
            "static_memory_bytes": 67633152.0,
            "static_memory_gib": 0.06298828125,
            "overall_peak_bytes": 353370112.0,
            "overall_peak_gib": 0.3291015625,
            "allocator_state": {
                "allocated_bytes": 218628096.0,
                "allocated_gib": 0.18798828125,
                "reserved_bytes": 218628096.0,
                "reserved_gib": 0.18798828125,
                "reserved_cached_bytes": 0.0,
                "reserved_cached_gib": 0.0,
            },
            "live_tensor_accounting": {
                "total_live_tensor_bytes": expected_linear_steady_state_bytes,
                "total_live_tensor_gib": 0.18798828125,
                "categories": {
                    "parameters": {"bytes": 67108864.0, "gib": 0.0625},
                    "gradients": {"bytes": 0.0, "gib": 0.0},
                    "optimizer_state": {"bytes": 134217728.0, "gib": 0.125},
                    "inputs": {"bytes": 262144.0, "gib": 0.000244140625},
                    "labels": {"bytes": 262144.0, "gib": 0.000244140625},
                    "other": {"bytes": 0.0, "gib": 0.0},
                },
                "top_other_tensors": [],
            },
            "annotation_memory": {
                "## forward ##_START": {
                    "annotation": {
                        "stage": "START",
                        "name": "## forward ##",
                        "device": 0,
                        "time_us": 0,
                    },
                    "memory_bytes": 151519232.0,
                    "memory_gib": 0.14111328125,
                },
                "## forward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## forward ##",
                        "device": 0,
                        "time_us": 1,
                    },
                    "memory_bytes": 151519232.0,
                    "memory_gib": 0.14111328125,
                },
                "## backward ##_END": {
                    "annotation": {
                        "stage": "END",
                        "name": "## backward ##",
                        "device": 0,
                        "time_us": 2,
                    },
                    "memory_bytes": 218628096.0,
                    "memory_gib": 0.20361328125,
                },
            },
            "files": {},
        }
    )

    report = compare_theory_to_measurement(theory, measured)

    assert report.metrics["static_baseline"].within_tolerance
    assert report.metrics["steady_state_floor"].within_tolerance
    assert report.metrics["backward_delta"].within_tolerance
    assert report.metrics["peak_over_floor"].within_tolerance
    assert report.metrics["dynamic_peak_lower_bound"].within_tolerance
    assert (
        report.memory_accounting.theoretical_model_memory_bytes
        == expected_linear_steady_state_bytes
    )
    assert report.memory_accounting.unexplained_gap_bytes == expected_non_tensor_gap_bytes
    assert report.memory_accounting.tensor_vs_theory_gap_bytes == 0.0
