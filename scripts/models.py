"""Model definitions used by the local comparison and benchmarking scripts."""

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from stella import StellaConfig

# nanoGPT is not a proper Python package (no __init__.py / pyproject.toml),
# so we add it to sys.path to import its model module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nanoGPT"))
from model import Block, GPTConfig, LayerNorm

# ── Base Transformer using nanoGPT blocks ────────────────────────────────────


class Transformer(nn.Module):
    """nanoGPT transformer backbone (continuous input, no embeddings)."""

    def __init__(
        self, n_embd: int = 8, n_head: int = 2, n_layer: int = 1, block_size: int = 16, **_: object
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformer blocks to a batch of continuous inputs."""
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


# ── LoRA Transformer (via PEFT) ──────────────────────────────────────────────


class LoRATransformer(nn.Module):
    """nanoGPT transformer with PEFT LoRA on attention & MLP projections."""

    def __init__(  # noqa: PLR0913
        self,
        n_embd: int = 8,
        n_head: int = 2,
        n_layer: int = 1,
        block_size: int = 16,
        rank: int = 4,
        alpha: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        base = Transformer(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size)
        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=["c_attn", "c_proj", "c_fc"], bias="none"
        )
        self.model = get_peft_model(cast("Any", base), lora_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LoRA-adapted model on a batch of inputs."""
        return self.model(x)


# ── StelLA Transformer (via official stella + PEFT) ──────────────────────────


class StelLATransformer(nn.Module):
    """nanoGPT transformer with StelLA adaptation on attention & MLP projections."""

    def __init__(  # noqa: PLR0913
        self,
        n_embd: int = 8,
        n_head: int = 2,
        n_layer: int = 1,
        block_size: int = 16,
        rank: int = 4,
        alpha: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        base = Transformer(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size)
        stella_config = StellaConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["c_attn", "c_proj", "c_fc"],
            bias="none",
            stella_grad_scaling=float(n_embd),
            stella_retraction="exp_map",
        )
        self.model = get_peft_model(cast("Any", base), stella_config)
        # Register the StellaModel so the optimizer hooks can find it.
        # This works because expressivity creates the model, then immediately
        # creates the optimizer from model.parameters().
        StelLAAdamW.set_current_stella_model(cast("_StellaHookModel", self.model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the StelLA-adapted model on a batch of inputs."""
        return self.model(x)


# ── Optimizer with Stiefel hooks for StelLA ──────────────────────────────────


class _StellaHookModel(Protocol):
    """Protocol for models exposing StelLA optimizer hook methods."""

    def pre_optimizer_step(self) -> None:
        """Run before the optimizer step."""

    def post_optimizer_step(self) -> None:
        """Run after the optimizer step."""


class StelLAAdamW(torch.optim.AdamW):
    """AdamW that registers StelLA's Riemannian pre/post hooks on creation.

    The expressivity library creates the optimizer as:
        optimizer = optimizer_class(model.parameters(), lr)
    We intercept this to find the StellaModel and register the hooks.
    """

    _current_stella_model: _StellaHookModel | None = None

    @classmethod
    def set_current_stella_model(cls, model: _StellaHookModel) -> None:
        """Store the current StelLA model so hooks can be attached during optimizer init."""
        cls._current_stella_model = model

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.001) -> None:
        super().__init__(list(params), lr=lr)
        stella_model = StelLAAdamW._current_stella_model
        if stella_model is not None:
            self.register_step_pre_hook(
                lambda _opt, _args, _kwargs: stella_model.pre_optimizer_step()
            )
            self.register_step_post_hook(
                lambda _opt, _args, _kwargs: stella_model.post_optimizer_step()
            )
