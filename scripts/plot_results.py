"""
Custom plots for expressivity results with shaded mean/min regions.
Reads JSON results from results/ and produces improved figures.

Produces:
  figures/direct_mean_comparison.png
  figures/direct_min_comparison.png
  figures/baseline_mean_comparison.png
  figures/baseline_min_comparison.png
  figures/ablation_mean_comparison.png
  figures/ablation_min_comparison.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
COLORS = {
    "LoRA": "#4C72B0",
    "StelLA": "#55A868",
    "Transformer": "#8172B2",
    "Euclidean 3-factor": "#C44E52",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_comparison(results, title_prefix, save_prefix, baseline_name=None):
    """Plot mean and min comparison with shaded regions between them."""
    A_name = results["A_space"]
    B_name = results["B_space"]

    meas_A = np.array(results["mesurements_A"])
    meas_B = np.array(results["mesurements_B"])
    mean_A = np.array(results["mean_A_fit"])
    mean_B = np.array(results["mean_B_fit"])
    min_A = np.array(results["min_A_fit"])
    min_B = np.array(results["min_B_fit"])

    ranks = list(range(1, len(meas_A) + 1))

    # ── Mean comparison (with shaded region to min) ──────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    # A
    ax.plot(meas_A, mean_A, "o-", color=COLORS.get(A_name, "#4C72B0"),
            label=f"{A_name} (mean)", linewidth=2, markersize=5)
    ax.fill_between(meas_A, min_A, mean_A,
                    color=COLORS.get(A_name, "#4C72B0"), alpha=0.15,
                    label=f"{A_name} (min–mean)")

    # B
    ax.plot(meas_B, mean_B, "s-", color=COLORS.get(B_name, "#55A868"),
            label=f"{B_name} (mean)", linewidth=2, markersize=5)
    ax.fill_between(meas_B, min_B, mean_B,
                    color=COLORS.get(B_name, "#55A868"), alpha=0.15,
                    label=f"{B_name} (min–mean)")

    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Fitting Loss (MSE)")
    ax.set_title(f"{title_prefix}: Mean Fitting Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add rank annotations
    for i, r in enumerate(ranks):
        ax.annotate(f"r={r}", (meas_A[i], mean_A[i]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color=COLORS.get(A_name, "#4C72B0"), ha="center")
        ax.annotate(f"r={r}", (meas_B[i], mean_B[i]),
                    textcoords="offset points", xytext=(0, -12),
                    fontsize=7, color=COLORS.get(B_name, "#55A868"), ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_mean_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Min comparison ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(meas_A, min_A, "o-", color=COLORS.get(A_name, "#4C72B0"),
            label=f"{A_name} (min)", linewidth=2, markersize=5)
    ax.plot(meas_B, min_B, "s-", color=COLORS.get(B_name, "#55A868"),
            label=f"{B_name} (min)", linewidth=2, markersize=5)

    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Best Fitting Loss (MSE)")
    ax.set_title(f"{title_prefix}: Best-case Fitting Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for i, r in enumerate(ranks):
        ax.annotate(f"r={r}", (meas_A[i], min_A[i]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color=COLORS.get(A_name, "#4C72B0"), ha="center")
        ax.annotate(f"r={r}", (meas_B[i], min_B[i]),
                    textcoords="offset points", xytext=(0, -12),
                    fontsize=7, color=COLORS.get(B_name, "#55A868"), ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_min_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    # Direct comparison
    direct = load_results("results/direct_comparison/results.json")
    plot_comparison(direct, "Direct: LoRA vs StelLA", "direct")

    # Baseline comparison
    baseline = load_results("results/transformer_baseline/results.json")
    plot_comparison(baseline, "Baseline: vs Full Transformer", "baseline")

    # Ablation (if available)
    ablation_path = "results/ablation/results.json"
    if os.path.exists(ablation_path):
        ablation = load_results(ablation_path)
        plot_comparison(ablation, "Ablation: Euclidean 3-factor vs StelLA", "ablation")
        print("Plotted ablation results")

    print("All plots saved to figures/")


if __name__ == "__main__":
    main()
