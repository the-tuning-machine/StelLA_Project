"""Model definitions used by the local comparison and benchmarking scripts."""

from collections.abc import Iterable
from typing import Any, Protocol, cast

import torch
from peft import LoraConfig, get_peft_model
from stella import StellaConfig
from torch import nn

from stellatscale._nanogpt_backbone import Block, GPTConfig, LayerNorm

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


class _SingleLinearBackbone(nn.Module):
    """Minimal wrapper exposing a named linear module for PEFT adaptation."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped linear layer."""
        return self.linear(x)


class LoRALinear(nn.Module):
    """Single linear layer adapted with the official PEFT LoRA wrapper."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int = 1) -> None:
        super().__init__()
        base.requires_grad_(requires_grad=False)
        backbone = _SingleLinearBackbone(base)
        lora_config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["linear"], bias="none")
        self.model = get_peft_model(cast("Any", backbone), lora_config)

    @property
    def base_layer(self) -> nn.Linear:
        """Return the underlying frozen dense projection."""
        base_model = cast("Any", self.model.base_model)
        return cast("nn.Linear", base_model.model.linear.base_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LoRA-adapted linear layer."""
        return self.model(x)


class StelLALinear(nn.Module):
    """Single linear layer adapted with the official PEFT StelLA wrapper."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int = 1) -> None:
        super().__init__()
        base.requires_grad_(requires_grad=False)
        backbone = _SingleLinearBackbone(base)
        stella_config = StellaConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["linear"],
            bias="none",
            stella_grad_scaling=float(base.out_features),
            stella_retraction="exp_map",
        )
        self.model = get_peft_model(cast("Any", backbone), stella_config)
        StelLAAdamW.set_current_stella_model(cast("_StellaHookModel", self.model))

    @property
    def base_layer(self) -> nn.Linear:
        """Return the underlying frozen dense projection."""
        base_model = cast("Any", self.model.base_model)
        return cast("nn.Linear", base_model.model.linear.base_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the StelLA-adapted linear layer."""
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
            stella_retraction="polar",
        )
        self.model = get_peft_model(cast("Any", base), stella_config)
        # Register the StellaModel so the optimizer hooks can find it.
        # This works because expressivity creates the model, then immediately
        # creates the optimizer from model.parameters().
        StelLAAdamW.set_current_stella_model(cast("_StellaHookModel", self.model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the StelLA-adapted model on a batch of inputs."""
        return self.model(x)


# ── Euclidean Three-Factor Transformer (ablation: same USV^T, no Stiefel) ────


class EuclideanThreeFactorTransformer(nn.Module):
    """Same three-factor decomposition USV^T as StelLA, but with standard AdamW.

    U and V start orthogonal but are free to drift during training.
    This isolates the effect of the Stiefel geometry from the parameterization.
    """

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
            stella_retraction="polar",
        )
        self.model = get_peft_model(cast("Any", base), stella_config)
        # NOTE: We do NOT set StelLAAdamW._current_stella_model here.
        # This model will be used with standard AdamW, so no hooks.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Euclidean three-factor model on a batch of inputs."""
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
    def set_current_stella_model(cls, model: _StellaHookModel | None) -> None:
        """Store the current StelLA model so hooks can be attached during optimizer init."""
        cls._current_stella_model = model

    @classmethod
    def clear_current_stella_model(cls) -> None:
        """Release the reference to the current StelLA model (avoids GPU memory leaks)."""
        cls._current_stella_model = None

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
