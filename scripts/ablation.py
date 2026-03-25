"""
Ablation: Euclidean 3-factor (USV^T, no Stiefel constraint) vs StelLA.
Isolates the effect of the Stiefel geometry from the parameterization.

Produces: results/ablation/results.json
"""
from expressivity import ArchitecturalSpace, ArchitectureComparator
from stellatscale.models import StelLATransformer, EuclideanThreeFactorTransformer, StelLAAdamW
import os

# ── Same hyper-parameters as main.py ─────────────────────────────────────────
N_EMBD = 8
N_HEAD = 2
BLOCK_SIZE = 16
N_LAYER = 1
INPUT_SIZE = (5, N_EMBD)
RANKS = [i + 1 for i in range(5)]
BATCH_SIZE = 256

base_kwargs = {"n_embd": N_EMBD, "n_head": N_HEAD, "block_size": BLOCK_SIZE, "n_layer": N_LAYER}

euclid_params = [{**base_kwargs, "rank": r} for r in RANKS]
stella_params = [{**base_kwargs, "rank": r} for r in RANKS]

# Euclidean 3-factor: same USV^T decomposition, standard AdamW (no Riemannian hooks)
euclid_space = ArchitecturalSpace(
    INPUT_SIZE, "Euclidean 3-factor", EuclideanThreeFactorTransformer, euclid_params,
    epoch=20, lr=0.01, automatic_mesurement_mode="parameters",
    batch_size=BATCH_SIZE, automatic_batch_size_scale=None,
)

# StelLA: USV^T with Stiefel constraint
stella_space = ArchitecturalSpace(
    INPUT_SIZE, "StelLA", StelLATransformer, stella_params,
    epoch=20, lr=0.01, automatic_mesurement_mode="parameters",
    batch_size=BATCH_SIZE, automatic_batch_size_scale=None,
    optimizer=StelLAAdamW,
)

# ── Run comparison ───────────────────────────────────────────────────────────
comparator = ArchitectureComparator(euclid_space, stella_space)
res = comparator.compare(
    1000,
    5,
    save_path=os.path.join(os.getcwd(), "results", "ablation"),
)
print(res)
