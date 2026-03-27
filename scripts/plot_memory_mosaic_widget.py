"""Interactive widget for Mosaic-aligned peak memory breakdowns."""

# %%
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.axes import Axes

from stellatscale.memory_experiment import (
    LinearModelVariant,
    MemoryExperimentConfig,
    build_theoretical_summary,
)

ROOT_DIR = Path(__file__).parents[1]
RESULTS_ROOT = ROOT_DIR / "results" / "memory" / "single_layer_lora"
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_SCRIPT = ROOT_DIR / "scripts" / "lora_memory_analysis.py"

CATEGORIES = ["Static", "Activation", "Backward", "Optimizer", "Unknown"]
TIMELINE_CATEGORIES = ["Static", "Activation", "Backward", "Optimizer", "Unknown"]
MODEL_NAMES = ["Linear", "LoRA", "StelLA"]
VARIANT_NAMES = ["linear", "linear_lora", "linear_stella"]
VARIANT_LABELS = {"Linear": "linear", "LoRA": "linear_lora", "StelLA": "linear_stella"}
COLORS = {
    "Static": "#355C7D",
    "Activation": "#90A955",
    "Backward": "#F8B195",
    "Optimizer": "#C3423F",
    "Unknown": "#D97D54",
}


def _config_tag(d_in: int, d_out: int, r: int, b: int) -> str:
    """Return the run-directory tag for one slider configuration."""
    return f"din_{d_in}_dout_{d_out}_r_{r}_b_{b}"


def _bytes_to_mib(num_bytes: float) -> float:
    """Convert bytes to mebibytes."""
    return num_bytes / 1024**2


def load_mosaic_peak_breakdown(
    d_in: int, d_out: int, r: int, b: int
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float, float] | None:
    """Load the exported Mosaic peak breakdown for one configuration, if present."""
    candidate_paths = [
        RESULTS_ROOT / "mosaic_peak_breakdown.json",
        RESULTS_ROOT / "runs" / _config_tag(d_in, d_out, r, b) / "mosaic_peak_breakdown.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if config.get("in_features") != d_in:
            continue
        if config.get("out_features") != d_out:
            continue
        if config.get("lora_rank") != r:
            continue
        if config.get("batch_size") != b:
            continue

        variants = payload["variants"]
        values: list[dict[str, float]] = []
        total_peak_values: list[float] = []
        for variant_name in VARIANT_NAMES:
            breakdown = variants[variant_name]
            categories = {
                category: _bytes_to_mib(float(breakdown["categories"][category]["bytes"]))
                for category in CATEGORIES
            }
            values.append(categories)
            total_peak_values.append(_bytes_to_mib(float(breakdown["total_peak_bytes"])))
        return (
            values[0],
            values[1],
            values[2],
            total_peak_values[0],
            total_peak_values[1],
            total_peak_values[2],
        )

    return None


def has_mosaic_peak_breakdown(d_in: int, d_out: int, r: int, b: int) -> bool:
    """Return whether a Mosaic peak export exists for the selected configuration."""
    return load_mosaic_peak_breakdown(d_in, d_out, r, b) is not None


def _mosaic_output_dir(d_in: int, d_out: int, r: int, b: int) -> Path | None:
    """Return the Mosaic output directory for the selected configuration, if present."""
    candidate_dirs = [
        RESULTS_ROOT / "mosaic",
        RESULTS_ROOT / "runs" / _config_tag(d_in, d_out, r, b) / "mosaic",
    ]
    for candidate_dir in candidate_dirs:
        summary_path = candidate_dir.parent / "mosaic_peak_breakdown.json"
        if not candidate_dir.exists() or not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if config.get("in_features") != d_in:
            continue
        if config.get("out_features") != d_out:
            continue
        if config.get("lora_rank") != r:
            continue
        if config.get("batch_size") != b:
            continue
        return candidate_dir
    return None


def mosaic_html_path(variant_name: str) -> Path | None:
    """Return the generated Mosaic categories HTML for the current widget configuration."""
    output_dir = _mosaic_output_dir(w_din.value, w_dout.value, w_r.value, w_b.value)
    if output_dir is None:
        return None
    candidate_path = output_dir / f"{variant_name}_categories.html"
    if candidate_path.exists():
        return candidate_path
    return None


def mosaic_summary_path(variant_name: str) -> Path | None:
    """Return the generated Mosaic summary JSON for the current widget configuration."""
    output_dir = _mosaic_output_dir(w_din.value, w_dout.value, w_r.value, w_b.value)
    if output_dir is None:
        return None
    candidate_path = output_dir / f"{variant_name}_summary.json"
    if candidate_path.exists():
        return candidate_path
    return None


def load_mosaic_categories_timeline(
    variant_name: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Load the Mosaic categories-over-time series from the generated categories HTML artifact."""
    html_path = mosaic_html_path(variant_name)
    if html_path is None:
        return None

    measured_breakdown = load_mosaic_peak_breakdown(w_din.value, w_dout.value, w_r.value, w_b.value)
    if measured_breakdown is None:
        return None
    variant_index = VARIANT_NAMES.index(variant_name)
    measured_variants = list(measured_breakdown[:3])
    static_mib = measured_variants[variant_index]["Static"]

    html_content = html_path.read_text(encoding="utf-8")
    spec_match = re.search(r"var spec = (\{.*?\});\s*var embedOpt", html_content, flags=re.DOTALL)
    if spec_match is None:
        return None

    spec = json.loads(spec_match.group(1))
    datasets = spec.get("datasets", {})
    if not datasets:
        return None

    records = next(iter(datasets.values()))
    event_indices = sorted({int(record["event_idx"]) for record in records})
    series_by_category: dict[str, dict[int, float]] = {
        category: dict.fromkeys(event_indices, 0.0) for category in TIMELINE_CATEGORIES
    }
    for event_idx in event_indices:
        series_by_category["Static"][event_idx] = static_mib
    for record in records:
        raw_category = str(record["cat"])
        _, normalized_category = raw_category.split("_", 1)
        category_name = normalized_category.title()
        if category_name not in series_by_category:
            continue
        series_by_category[category_name][int(record["event_idx"])] += float(record["sum"]) * 1024.0

    x_values = np.array(event_indices, dtype=float)
    y_values = {
        category: np.array(
            [series_by_category[category][event_idx] for event_idx in event_indices], dtype=float
        )
        for category in TIMELINE_CATEGORIES
    }
    return x_values, y_values


def load_mosaic_step_time_ms(variant_name: str) -> float | None:
    """Return the average measured step duration in milliseconds for one variant."""
    summary_path = mosaic_summary_path(variant_name)
    if summary_path is None:
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    annotation_memory = payload.get("annotation_memory", {})
    durations_ms: list[float] = []
    for occurrence in range(1, 128):
        suffix = "" if occurrence == 1 else f"({occurrence - 1})"
        start_key = f"## forward ##_START{suffix}"
        end_key = f"## optimizer ##_END{suffix}"
        start_payload = annotation_memory.get(start_key)
        end_payload = annotation_memory.get(end_key)
        if start_payload is None or end_payload is None:
            if occurrence > 1:
                break
            continue
        start_time = float(start_payload["annotation"]["time_us"])
        end_time = float(end_payload["annotation"]["time_us"])
        durations_ms.append((end_time - start_time) / 1000.0)

    if not durations_ms:
        return None
    return float(sum(durations_ms) / len(durations_ms))


def _analytical_variant_breakdown(
    config: MemoryExperimentConfig, variant: LinearModelVariant
) -> tuple[dict[str, float], float]:
    """Build a runtime-aligned analytical breakdown for one variant."""
    theory = build_theoretical_summary(config, variant)
    categories = {
        "Static": _bytes_to_mib(float(theory.static_baseline_bytes)),
        "Activation": _bytes_to_mib(float(theory.forward_dynamic_estimate_bytes)),
        "Backward": _bytes_to_mib(float(theory.gradient_bytes)),
        "Optimizer": _bytes_to_mib(float(theory.optimizer_state_bytes)),
        "Unknown": 0.0,
    }
    return categories, sum(categories.values())


def load_analytical_breakdown(
    d_in: int, d_out: int, r: int, b: int
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float, float]:
    """Build the analytical prediction using the runtime-aligned bucket semantics."""
    config = MemoryExperimentConfig(batch_size=b, in_features=d_in, out_features=d_out, lora_rank=r)
    linear_categories, linear_total_peak = _analytical_variant_breakdown(
        config, LinearModelVariant.LINEAR
    )
    lora_categories, lora_total_peak = _analytical_variant_breakdown(
        config, LinearModelVariant.LINEAR_LORA
    )
    stella_categories, stella_total_peak = _analytical_variant_breakdown(
        config, LinearModelVariant.LINEAR_STELLA
    )
    return (
        linear_categories,
        lora_categories,
        stella_categories,
        linear_total_peak,
        lora_total_peak,
        stella_total_peak,
    )


style = {"description_width": "initial"}
layout = widgets.Layout(width="400px")
w_din = widgets.IntSlider(
    value=4096,
    min=128,
    max=16384,
    step=128,
    description="Input Dim (d_in):",
    style=style,
    layout=layout,
)
w_dout = widgets.IntSlider(
    value=4096,
    min=128,
    max=16384,
    step=128,
    description="Output Dim (d_out):",
    style=style,
    layout=layout,
)
w_r = widgets.IntSlider(
    value=128, min=1, max=256, step=1, description="LoRA Rank (r):", style=style, layout=layout
)
w_b = widgets.IntSlider(
    value=128, min=1, max=512, step=1, description="Batch Size (b):", style=style, layout=layout
)
btn_run_experiment = widgets.Button(description="▶ Run Experiment", button_style="warning")
btn_save_png = widgets.Button(description="💾 Save PNG", button_style="success")
out_plot = widgets.Output()
status_message = widgets.HTML(value="")


def _config_text() -> str:
    """Return the current configuration label used in the composite figure footer."""
    return f"d_in: {w_din.value}  |  d_out: {w_dout.value}  |  Rank (r): {w_r.value}  |  Batch (b): {w_b.value}"


def _load_composite_plot_data() -> tuple[
    list[dict[str, float]],
    list[float],
    list[dict[str, float]],
    list[float],
    list[tuple[np.ndarray, dict[str, np.ndarray]]],
]:
    """Load the measured and analytical breakdowns plus all measured timelines."""
    measured_breakdown = load_mosaic_peak_breakdown(w_din.value, w_dout.value, w_r.value, w_b.value)
    if measured_breakdown is None:
        message = "No Mosaic peak breakdown exists for the selected configuration."
        raise ValueError(message)

    analytical_breakdown = load_analytical_breakdown(
        w_din.value, w_dout.value, w_r.value, w_b.value
    )
    measured_values = list(measured_breakdown[:3])
    measured_totals = list(measured_breakdown[3:])
    analytical_values = list(analytical_breakdown[:3])
    analytical_totals = list(analytical_breakdown[3:])

    timeline_payloads: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for variant_name in VARIANT_NAMES:
        timeline = load_mosaic_categories_timeline(variant_name)
        if timeline is None:
            message = f"No generated Mosaic categories timeline exists for {variant_name}. Run the experiment first."
            raise ValueError(message)
        timeline_payloads.append(timeline)

    return measured_values, measured_totals, analytical_values, analytical_totals, timeline_payloads


def _scale_breakdown_values(
    measured_values: list[dict[str, float]],
    measured_totals: list[float],
    analytical_values: list[dict[str, float]],
    analytical_totals: list[float],
) -> tuple[list[dict[str, float]], list[float], list[dict[str, float]], list[float]]:
    """Scale only the top breakdown stacks to percentages of the measured dense baseline."""
    baseline_total = measured_totals[0]
    measured_values = [
        {category: value * (100.0 / baseline_total) for category, value in breakdown.items()}
        for breakdown in measured_values
    ]
    analytical_values = [
        {category: value * (100.0 / baseline_total) for category, value in breakdown.items()}
        for breakdown in analytical_values
    ]
    return measured_values, measured_totals, analytical_values, analytical_totals


def _draw_breakdown_axis(
    ax_breakdown: Axes,
    measured_values: list[dict[str, float]],
    measured_totals: list[float],
    analytical_values: list[dict[str, float]],
    analytical_totals: list[float],
) -> None:
    """Draw the combined measured-vs-analytical stacked breakdown panel."""
    model_centers = np.array(range(len(MODEL_NAMES)), dtype=float) * 2.4
    bar_width = 0.84
    source_offsets = {"Measured": -0.42, "Analytical": 0.42}
    breakdown_sets = {
        "Measured": (measured_values, measured_totals),
        "Analytical": (analytical_values, analytical_totals),
    }

    all_bar_totals = [
        *(sum(breakdown.values()) for breakdown in measured_values),
        *(sum(breakdown.values()) for breakdown in analytical_values),
    ]
    max_bar_total = max(all_bar_totals)
    label_offset = max(0.02 * max_bar_total, 1.2)
    for source_name, (breakdowns, totals) in breakdown_sets.items():
        x_positions = model_centers + source_offsets[source_name]
        bottoms = np.zeros(len(MODEL_NAMES))
        for category in CATEGORIES:
            heights = np.array([breakdown[category] for breakdown in breakdowns])
            ax_breakdown.bar(
                x_positions,
                heights,
                bottom=bottoms,
                width=bar_width,
                color=COLORS[category],
                edgecolor="white",
                linewidth=1.2,
            )
            bottoms += heights
        for index, total in enumerate(totals):
            ax_breakdown.text(
                x_positions[index],
                bottoms[index] + label_offset,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="#1F2933",
            )

    for center in model_centers:
        ax_breakdown.text(
            center + source_offsets["Measured"],
            -max_bar_total * 0.06,
            "Measured",
            ha="center",
            va="top",
            fontsize=11,
            color="#51606F",
        )
        ax_breakdown.text(
            center + source_offsets["Analytical"],
            -max_bar_total * 0.06,
            "Analytical",
            ha="center",
            va="top",
            fontsize=11,
            color="#51606F",
        )

    ax_breakdown.set_xlim(model_centers[0] - 1.3, model_centers[-1] + 1.3)
    ax_breakdown.set_ylim(-max_bar_total * 0.12, max_bar_total * 1.15)
    ax_breakdown.set_xticks([])
    ax_breakdown.set_ylabel("Memory (%)", fontsize=16)
    ax_breakdown.tick_params(axis="y", labelsize=14)
    ax_breakdown.spines["top"].set_visible(False)
    ax_breakdown.spines["right"].set_visible(False)
    ax_breakdown.spines["left"].set_visible(False)
    ax_breakdown.spines["bottom"].set_visible(False)
    ax_breakdown.yaxis.grid(visible=True, linestyle="--", alpha=0.5)
    ax_breakdown.set_axisbelow(True)
    ax_breakdown.set_facecolor("#FAFAF7")

    handles = [
        mpatches.Patch(facecolor=COLORS[category], edgecolor="white") for category in CATEGORIES
    ]
    ax_breakdown.legend(
        handles,
        CATEGORIES,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=5,
        frameon=False,
        fontsize=15,
        handletextpad=0.7,
        columnspacing=1.1,
    )
    ax_breakdown.text(
        0.985,
        0.985,
        _config_text(),
        transform=ax_breakdown.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        family="monospace",
        color="#51606F",
        bbox={"facecolor": "#FFFFFFCC", "edgecolor": "#D7DBE0", "boxstyle": "round,pad=0.35"},
    )


def _draw_timeline_axis(
    axis: Axes,
    variant_name: str,
    timeline: tuple[np.ndarray, dict[str, np.ndarray]],
    *,
    show_ylabel: bool,
) -> None:
    """Draw one measured Mosaic timeline subplot."""
    x_values, y_values = timeline
    stacked_series = [y_values[category] for category in TIMELINE_CATEGORIES]

    axis.stackplot(
        x_values,
        stacked_series,
        colors=[COLORS[category] for category in TIMELINE_CATEGORIES],
        alpha=0.85,
    )
    total_series = np.sum(np.vstack(stacked_series), axis=0)
    peak_index = int(np.argmax(total_series))
    axis.axvline(x_values[peak_index], color="#1F2933", linestyle="--", linewidth=1.6, alpha=0.9)
    axis.set_xlabel("")
    if show_ylabel:
        axis.set_ylabel("Memory (MiB)", fontsize=16)
    else:
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines["left"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.grid(visible=True, linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    axis.set_facecolor("#FAFAF7")

    step_time_ms = load_mosaic_step_time_ms(variant_name)
    if step_time_ms is not None:
        axis.text(
            0.98,
            0.96,
            f"avg step {step_time_ms:.2f} ms",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="#51606F",
            bbox={"facecolor": "#FFFFFFCC", "edgecolor": "#D7DBE0", "boxstyle": "round,pad=0.25"},
        )


def draw_plot(*, is_saving: bool = False, filepath: Path | None = None) -> None:
    """Render one composite figure with breakdowns and all Mosaic timelines."""
    measured_values, measured_totals, analytical_values, analytical_totals, timeline_payloads = (
        _load_composite_plot_data()
    )
    measured_values, measured_totals, analytical_values, analytical_totals = (
        _scale_breakdown_values(
            measured_values, measured_totals, analytical_values, analytical_totals
        )
    )

    fig = plt.figure(figsize=(16, 14), dpi=100 if not is_saving else 300)
    fig.patch.set_facecolor("#FFFFFF")
    grid = fig.add_gridspec(3, 3, height_ratios=[2.05, 2.6, 0.14], hspace=0.0, wspace=0.0)
    ax_breakdown = fig.add_subplot(grid[0, :])
    first_timeline_axis = fig.add_subplot(grid[1, 0])
    timeline_axes = [
        first_timeline_axis,
        fig.add_subplot(grid[1, 1], sharey=first_timeline_axis),
        fig.add_subplot(grid[1, 2], sharey=first_timeline_axis),
    ]
    _draw_breakdown_axis(
        ax_breakdown, measured_values, measured_totals, analytical_values, analytical_totals
    )
    for index, (axis, variant_name, timeline) in enumerate(
        zip(timeline_axes, VARIANT_NAMES, timeline_payloads, strict=True)
    ):
        _draw_timeline_axis(axis, variant_name, timeline, show_ylabel=index == 0)

    for axis, model_name in zip(timeline_axes, MODEL_NAMES, strict=True):
        bounds = axis.get_position()
        fig.text(
            (bounds.x0 + bounds.x1) / 2,
            bounds.y0 - 0.018,
            model_name,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="#1F2933",
        )
    if is_saving and filepath is not None:
        fig.savefig(filepath, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def update(*args: Any) -> None:
    """Refresh the figure after any widget change."""
    if not has_mosaic_peak_breakdown(w_din.value, w_dout.value, w_r.value, w_b.value):
        status_message.value = "<span style='color:#8a5a00'>No Mosaic peak export exists for this configuration yet. Run the experiment first.</span>"
    else:
        status_message.value = "<span style='color:#1b7f3b'>Showing combined measured Mosaic, analytical breakdown, and all three Mosaic timelines.</span>"

    with out_plot:
        clear_output(wait=True)
        if has_mosaic_peak_breakdown(w_din.value, w_dout.value, w_r.value, w_b.value):
            draw_plot()


def run_experiment(*args: Any) -> None:
    """Run the profiling workflow and regenerate the Mosaic peak export."""
    btn_run_experiment.disabled = True
    btn_save_png.disabled = True
    status_message.value = (
        "<span style='color:#8a5a00'>Running experiment for the selected configuration...</span>"
    )
    command = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        "--in-features",
        str(w_din.value),
        "--out-features",
        str(w_dout.value),
        "--lora-rank",
        str(w_r.value),
        "--batch-size",
        str(w_b.value),
    ]
    try:
        subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        error_output = exc.stderr.strip() or exc.stdout.strip() or "Unknown error"
        status_message.value = f"<span style='color:#b42318'>Experiment failed.</span><pre style='white-space:pre-wrap'>{error_output}</pre>"
    else:
        status_message.value = "<span style='color:#1b7f3b'>Experiment finished. Mosaic peak breakdown loaded for this configuration.</span>"
        update()
    finally:
        btn_run_experiment.disabled = False
        btn_save_png.disabled = False


def save_png(*args: Any) -> None:
    """Save the current composite Mosaic figure as a PNG image."""
    filepath = FIGURES_DIR / (
        f"single_layer_memory_mosaic_widget_din_{w_din.value}_dout_{w_dout.value}"
        f"_r_{w_r.value}_b_{w_b.value}.png"
    )
    draw_plot(is_saving=True, filepath=filepath)
    status_message.value = f"<span style='color:#1b7f3b'>Saved PNG to {filepath}</span>"


for widget in [w_din, w_dout, w_r, w_b]:
    widget.observe(update, names="value")
btn_run_experiment.on_click(run_experiment)
btn_save_png.on_click(save_png)

sliders_col1 = widgets.VBox([w_din, w_dout])
sliders_col2 = widgets.VBox([w_r, w_b])
controls = widgets.HBox([sliders_col1, sliders_col2])
buttons = widgets.HBox([btn_run_experiment, btn_save_png])
ui_layout = widgets.VBox(
    [controls, buttons, status_message],
    layout=widgets.Layout(align_items="center", margin="20px 0 0 0"),
)

display(out_plot, ui_layout)
update()
