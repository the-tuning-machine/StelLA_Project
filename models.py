import sys
import os

import torch
import torch.nn as nn

# nanoGPT is not a proper Python package (no pyproject.toml / setup.py),
# so we add it to sys.path to import its model module.
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nanoGPT"))
from nanoGPT.model import GPTConfig, Block, LayerNorm

# stella is installed via: uv add "stella @ git+https://github.com/SonyResearch/stella"
from stella import StellaConfig

# peft is installed via: uv add peft
from peft import LoraConfig, get_peft_model


# ── Base Transformer using nanoGPT blocks ────────────────────────────────────


class Transformer(nn.Module):
    """nanoGPT transformer backbone (continuous input, no embeddings)."""

    def __init__(self, n_embd=8, n_head=2, n_layer=1, block_size=16, **kwargs):
        super().__init__()
        config = GPTConfig(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            vocab_size=1,  # unused, required by GPTConfig
            bias=False,
            dropout=0.0,
        )
        self.blocks = nn.ModuleList([Block(config) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


# ── LoRA Transformer (via PEFT) ──────────────────────────────────────────────


class LoRATransformer(nn.Module):
    """nanoGPT transformer with PEFT LoRA on attention & MLP projections."""

    def __init__(self, n_embd=8, n_head=2, n_layer=1, block_size=16, rank=4, alpha=1, **kwargs):
        super().__init__()
        base = Transformer(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["c_attn", "c_proj", "c_fc"],
            bias="none",
        )
        self.model = get_peft_model(base, lora_config)

    def forward(self, x):
        return self.model(x)


# ── StelLA Transformer (via official stella + PEFT) ──────────────────────────


class StelLATransformer(nn.Module):
    """nanoGPT transformer with StelLA adaptation on attention & MLP projections."""

    def __init__(self, n_embd=8, n_head=2, n_layer=1, block_size=16, rank=4, alpha=1, **kwargs):
        super().__init__()
        base = Transformer(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size)
        stella_config = StellaConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["c_attn", "c_proj", "c_fc"],
            bias="none",
        )
        self.model = get_peft_model(base, stella_config)

    def forward(self, x):
        return self.model(x)
