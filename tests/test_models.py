"""Tests for the packaged model definitions."""

import torch
from pytest_mock import MockerFixture
from torch import nn

from stellatscale import models
from stellatscale._nanogpt_backbone import Block, GPTConfig, LayerNorm


def test_transformer_forward_shape() -> None:
    """The base transformer should preserve the batch, sequence, and embedding shape."""
    model = models.Transformer(n_embd=8, n_head=2, n_layer=1, block_size=16)
    inputs = torch.randn(2, 5, 8)

    outputs = model(inputs)

    assert outputs.shape == inputs.shape


def test_adapter_models_forward_with_mocked_peft(mocker: MockerFixture) -> None:
    """The LoRA and StelLA wrappers should delegate to the wrapped base model."""
    mocker.patch("stellatscale.models.get_peft_model", side_effect=lambda base, _cfg: base)
    set_current = mocker.patch.object(models.StelLAAdamW, "set_current_stella_model")
    inputs = torch.randn(2, 5, 8)

    lora_model = models.LoRATransformer(n_embd=8, n_head=2, n_layer=1, block_size=16, rank=2)
    stella_model = models.StelLATransformer(n_embd=8, n_head=2, n_layer=1, block_size=16, rank=2)

    assert lora_model(inputs).shape == inputs.shape
    assert stella_model(inputs).shape == inputs.shape
    set_current.assert_called_once()


def test_stella_optimizer_hooks_execute() -> None:
    """The custom optimizer should run the registered StelLA hooks around a step."""

    class HookModel:
        def __init__(self) -> None:
            self.pre_calls = 0
            self.post_calls = 0

        def pre_optimizer_step(self) -> None:
            self.pre_calls += 1

        def post_optimizer_step(self) -> None:
            self.post_calls += 1

    hook_model = HookModel()
    parameter = nn.Parameter(torch.tensor(1.0))
    models.StelLAAdamW.set_current_stella_model(hook_model)
    optimizer = models.StelLAAdamW([parameter], lr=0.1)

    loss = parameter.square()
    loss.backward()
    optimizer.step()

    assert hook_model.pre_calls == 1
    assert hook_model.post_calls == 1


def test_internalized_nanogpt_block_preserves_shape() -> None:
    """The internalized nanoGPT-derived block should preserve tensor shape."""
    config = GPTConfig(n_embd=8, n_head=2, n_layer=1, block_size=16, dropout=0.0, bias=False)
    block = Block(config)
    inputs = torch.randn(2, 5, 8)

    outputs = block(inputs)

    assert outputs.shape == inputs.shape


def test_internalized_layer_norm_without_bias() -> None:
    """The copied LayerNorm should support the bias-free configuration used in the project."""
    layer_norm = LayerNorm(8, bias=False)
    inputs = torch.randn(2, 5, 8)

    outputs = layer_norm(inputs)

    assert outputs.shape == inputs.shape
    assert layer_norm.bias is None
