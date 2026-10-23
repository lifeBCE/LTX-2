"""int8-cast for LoRA training: bit-identical weights to quanto's int8, a frozen layer that computes like its
dequantized reference and passes gradients to inputs only, fp32 PEFT adapters, and the loader-only contract."""

import pytest
import torch
from optimum.quanto import freeze, qint8, quantize
from peft import LoraConfig, get_peft_model
from torch import nn

from ltx_core.quantization.int8_cast import (
    RETYPE_INT8_CAST,
    TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP,
    Int8CastLinear,
    dequantize_int8_per_row,
    quantize_weight_int8_per_row,
)
from ltx_trainer.quantization import prepare_int8_cast_linears_for_lora, quantize_model

IN_FEATURES, OUT_FEATURES, ROWS = 64, 32, 5
KEY = "transformer_blocks.0.attn1.to_q.weight"


@pytest.mark.parametrize(("out_features", "in_features"), [(4096, 4096), (2048, 4096), (512, 16384)])
def test_quantization_matches_quanto_bit_for_bit(out_features: int, in_features: int) -> None:
    torch.manual_seed(1)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02
    data, scale = quantize_weight_int8_per_row(weight)

    reference = nn.Sequential(nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16))
    reference[0].weight.data.copy_(weight)
    quantize(reference, weights=qint8)
    freeze(reference)
    quanto_weight = reference[0].weight

    assert torch.equal(data, quanto_weight._data)
    assert torch.equal(scale, quanto_weight._scale.squeeze(1))
    assert torch.equal(
        dequantize_int8_per_row(data, scale, torch.bfloat16),
        quanto_weight._data.to(torch.bfloat16) * quanto_weight._scale.to(torch.bfloat16),
    )


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _Attention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn1.to_q(x)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_blocks[0](x)


def _int8_model() -> tuple[_Model, torch.Tensor, torch.Tensor]:
    """A model loaded the way the policy loads it, plus the dequantized weight and the bias for reference."""
    torch.manual_seed(0)
    model = _Model()
    layer = model.transformer_blocks[0].attn1.to_q
    weight, bias = layer.weight.detach().clone(), layer.bias.detach().clone()
    RETYPE_INT8_CAST.mutator(model)
    sd = {
        r.new_key.removeprefix("transformer_blocks.0.attn1.to_q."): r.new_value
        for r in TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP.apply_to_key_value(KEY, weight)
    }
    sd["bias"] = bias
    layer.load_state_dict(sd, assign=True)
    model.requires_grad_(False)
    return model, dequantize_int8_per_row(sd["weight"], sd["weight_scale"], torch.bfloat16), bias


def test_int8_linear_matches_dequantized_and_grads_only_inputs() -> None:
    model, weight, bias = _int8_model()
    layer = model.transformer_blocks[0].attn1.to_q
    assert prepare_int8_cast_linears_for_lora(model) == 1
    assert isinstance(layer, Int8CastLinear)
    assert type(layer).__name__ == "Int8CastLinearTrain"
    assert layer.compute_dtype == torch.bfloat16
    assert layer.weight.dtype == torch.int8

    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
    out = model(x)
    reference_x = x.detach().requires_grad_()
    reference = nn.functional.linear(reference_x, weight, bias)
    assert torch.equal(out, reference)

    out.float().square().sum().backward()
    reference.float().square().sum().backward()
    assert torch.equal(x.grad, reference_x.grad)
    assert layer.weight.grad is None
    assert layer.weight_scale.grad is None


def test_peft_adapters_are_fp32() -> None:
    model, _, _ = _int8_model()
    prepare_int8_cast_linears_for_lora(model)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=["to_q"], init_lora_weights=True))
    lora_params = {name: p for name, p in peft_model.named_parameters() if "lora_" in name}
    assert lora_params
    assert all(p.dtype == torch.float32 for p in lora_params.values())
    peft_model(torch.randn(ROWS, IN_FEATURES, dtype=torch.bfloat16)).float().sum().backward()
    assert all(p.grad is not None for p in lora_params.values())
    assert all(torch.isfinite(p.grad).all() for p in lora_params.values())


def test_quantize_model_rejects_int8_cast() -> None:
    with pytest.raises(ValueError, match="int8_cast_policy"):
        quantize_model(nn.Linear(4, 4), precision="int8-cast")
