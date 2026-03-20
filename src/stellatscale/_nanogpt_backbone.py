"""Minimal transformer backbone adapted from nanoGPT by Andrej Karpathy.

This module internalizes the small subset of nanoGPT used by stellatscale:
LayerNorm, GPTConfig, CausalSelfAttention, MLP, and Block.

Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
Copyright (c) 2022 Andrej Karpathy
Licensed under the MIT License. See THIRD_PARTY_NOTICES.md for the retained
license text and attribution.
"""

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias parameter."""

    def __init__(self, ndim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization with a fixed epsilon for numerical stability."""
        return functional.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class GPTConfig:
    """Configuration for the minimal GPT-style transformer block stack."""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention used inside the transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            msg = "Embedding dimension must be divisible by the number of attention heads"
            raise ValueError(msg)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention over a batch of sequence embeddings."""
        batch_size, sequence_length, embed_dim = inputs.size()

        query, key, value = self.c_attn(inputs).split(self.n_embd, dim=2)
        key = key.view(
            batch_size, sequence_length, self.n_head, embed_dim // self.n_head
        ).transpose(1, 2)
        query = query.view(
            batch_size, sequence_length, self.n_head, embed_dim // self.n_head
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.n_head, embed_dim // self.n_head
        ).transpose(1, 2)

        if self.flash:
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            causal_mask = cast("torch.Tensor", self.causal_mask)
            attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            attention = attention.masked_fill(
                causal_mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
            )
            attention = functional.softmax(attention, dim=-1)
            attention = self.attn_dropout(attention)
            attended = attention @ value

        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embed_dim)
        )
        return self.resid_dropout(self.c_proj(attended))


class MLP(nn.Module):
    """Feed-forward block used in each transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the transformer feed-forward network."""
        outputs = self.c_fc(inputs)
        outputs = self.gelu(outputs)
        outputs = self.c_proj(outputs)
        return self.dropout(outputs)


class Block(nn.Module):
    """Single pre-norm transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP residual blocks."""
        outputs = inputs + self.attn(self.ln_1(inputs))
        return outputs + self.mlp(self.ln_2(outputs))
