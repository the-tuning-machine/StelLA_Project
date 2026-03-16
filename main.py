from expressivity import ArchitecturalSpace, ArchitectureComparator
from models import LoRATransformer, StelLATransformer, StelLAAdamW, Transformer

# ── Shared architecture hyper-parameters (tiny) ─────────────────────────────

N_EMBD = 8
N_HEAD = 2
BLOCK_SIZE = 16
N_LAYER = 1
INPUT_SIZE = (5, N_EMBD)  # (seq_len, n_embd)
RANKS = [i + 1 for i in range(5)]

# ── Parameter grids ──────────────────────────────────────────────────────────

base_kwargs = {"n_embd": N_EMBD, "n_head": N_HEAD, "block_size": BLOCK_SIZE, "n_layer": N_LAYER}

lora_params = [{**base_kwargs, "rank": r} for r in RANKS]
stella_params = [{**base_kwargs, "rank": r} for r in RANKS]
transformer_params = [base_kwargs for _ in RANKS]

# ── Architectural spaces ─────────────────────────────────────────────────────

BATCH_SIZE = 256

lora_space = ArchitecturalSpace(
    INPUT_SIZE, "LoRA", LoRATransformer, lora_params,
    epoch=10, lr=0.01, automatic_mesurement_mode="parameters",
    batch_size=BATCH_SIZE, automatic_batch_size_scale=None,
)

stella_space = ArchitecturalSpace(
    INPUT_SIZE, "StelLA", StelLATransformer, stella_params,
    epoch=10, lr=0.01, automatic_mesurement_mode="parameters",
    batch_size=BATCH_SIZE, automatic_batch_size_scale=None,
    optimizer=StelLAAdamW,
)

transformer_space = ArchitecturalSpace(
    INPUT_SIZE, "Transformer", Transformer, transformer_params,
    epoch=10, lr=0.01, automatic_mesurement_mode="parameters",
    batch_size=BATCH_SIZE, automatic_batch_size_scale=None,
)

# ── Comparisons ──────────────────────────────────────────────────────────────

# 1. LoRA vs StelLA (direct)
comparator_direct = ArchitectureComparator(lora_space, stella_space)
res1 = comparator_direct.compare(20, 20)
print(res1)
comparator_direct.plot("min")

# 2. LoRA vs StelLA with Transformer as neutral baseline
comparator_baseline = ArchitectureComparator(lora_space, stella_space, transformer_space)
res2 = comparator_baseline.compare(20, 20)
print(res2)
comparator_baseline.plot("min")
