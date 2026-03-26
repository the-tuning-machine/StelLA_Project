"""
Empirical verification that StelLA maintains orthogonality during training,
while the Euclidean three-factor baseline drifts.

Produces: figures/orthogonality_drift.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from stellatscale.models import EuclideanThreeFactorTransformer, StelLAAdamW, StelLATransformer

# ── Config ───────────────────────────────────────────────────────────────────
N_EMBD = 8
N_HEAD = 2
N_LAYER = 1
BLOCK_SIZE = 16
RANK = 4
STEPS = 200
LR = 0.01
BATCH = 64
SEQ_LEN = 5


def get_stiefel_matrices(model):
    """Extract all U and Vt matrices from a PEFT StelLA/Euclidean model."""
    mats = []
    for name, param in model.named_parameters():
        if ("stella_U" in name or "stella_Vt" in name) and "weight" in name:
            mats.append(param.data.clone())
    return mats


def orthogonality_deviation(mats):
    """Compute mean ||M M^T - I||_F across all Stiefel matrices.
    M is (out_dim, r), so M^T M should be I_r.
    """
    devs = []
    for M in mats:
        # M shape: (out_features, r) -- columns should be orthonormal
        r = min(M.shape)
        if M.shape[0] >= M.shape[1]:
            # tall matrix: M^T M = I_r
            dev = torch.norm(M.T @ M - torch.eye(r, device=M.device)).item()
        else:
            # wide matrix: M M^T = I_r
            dev = torch.norm(M @ M.T - torch.eye(r, device=M.device)).item()
        devs.append(dev)
    return sum(devs) / len(devs)


def train_and_track(name, model, optimizer, steps):
    """Train on random data and track orthogonality at each step."""
    criterion = nn.MSELoss()
    x = torch.randn(BATCH, SEQ_LEN, N_EMBD)
    target = torch.randn(BATCH, SEQ_LEN, N_EMBD)

    deviations = []
    for step in range(steps):
        # Measure before step
        mats = get_stiefel_matrices(model)
        if mats:
            deviations.append(orthogonality_deviation(mats))

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    # Final measurement
    mats = get_stiefel_matrices(model)
    if mats:
        deviations.append(orthogonality_deviation(mats))

    print(f"{name}: init dev={deviations[0]:.6f}, final dev={deviations[-1]:.6f}")
    return deviations


def main():
    torch.manual_seed(42)
    base_kw = {
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "block_size": BLOCK_SIZE,
        "rank": RANK,
    }

    # StelLA (Stiefel constraint)
    torch.manual_seed(42)
    m_stella = StelLATransformer(**base_kw)
    opt_stella = StelLAAdamW(m_stella.parameters(), lr=LR)
    devs_stella = train_and_track("StelLA", m_stella, opt_stella, STEPS)

    # Euclidean 3-factor (no constraint)
    torch.manual_seed(42)
    m_euclid = EuclideanThreeFactorTransformer(**base_kw)
    opt_euclid = torch.optim.AdamW(m_euclid.parameters(), lr=LR)
    devs_euclid = train_and_track("Euclidean", m_euclid, opt_euclid, STEPS)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(devs_stella, label="StelLA (Stiefel)", color="#55A868", linewidth=2)
    ax.plot(devs_euclid, label="Euclidean 3-factor", color="#C44E52", linewidth=2, linestyle="--")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\| U^\top U - I_r \|_F$ (mean over adapters)")
    ax.set_title("Orthogonality Deviation During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/orthogonality_drift.png", dpi=150)
    plt.close(fig)
    print("Saved figures/orthogonality_drift.png")


if __name__ == "__main__":
    main()
