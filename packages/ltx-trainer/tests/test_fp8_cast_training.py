"""fp8-cast base for LoRA training: the float8 Linears compute like their bf16 upcast, pass gradients to their
inputs only, give PEFT bf16/fp32 adapters, and quantize_model refuses the option it does not own."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.fp8_cast import UPCAST_DURING_INFERENCE, Fp8CastLinear
from ltx_trainer.model_loader import with_quantization_policy
from ltx_trainer.quantization import prepare_fp8_cast_linears_for_lora, quantize_model

IN_FEATURES, OUT_FEATURES, ROWS = 64, 32, 5


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
    """Mimics the LTX transformer's ``transformer_blocks.N.attn1.to_q`` naming that the fp8-cast policy matches."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_blocks[0](x)


def _fp8_model() -> tuple[_Model, torch.Tensor, torch.Tensor]:
    """A model whose ``to_q`` holds float8 weights, plus the bf16 weight and bias those float8 values round to."""
    torch.manual_seed(0)
    model = _Model()
    layer = model.transformer_blocks[0].attn1.to_q
    weight8 = layer.weight.detach().to(torch.float8_e4m3fn)
    bias8 = layer.bias.detach().to(torch.float8_e4m3fn)
    UPCAST_DURING_INFERENCE.mutator(model)  # retypes to Fp8CastLinear with empty float8 parameters
    layer.weight.data, layer.bias.data = weight8, bias8
    model.requires_grad_(False)
    return model, weight8.to(torch.bfloat16), bias8.to(torch.bfloat16)


def test_fp8_linear_matches_upcast_and_grads_only_inputs() -> None:
    model, weight, bias = _fp8_model()
    layer = model.transformer_blocks[0].attn1.to_q
    assert prepare_fp8_cast_linears_for_lora(model) == 1
    assert isinstance(layer, Fp8CastLinear)
    assert type(layer).__name__ == "Fp8CastLinearTrain"
    assert layer.compute_dtype == torch.bfloat16
    assert layer.weight.dtype == torch.float8_e4m3fn

    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
    out = model(x)
    reference_x = x.detach().requires_grad_()
    reference = nn.functional.linear(reference_x, weight, bias)
    assert torch.equal(out, reference)

    out.float().square().sum().backward()
    reference.float().square().sum().backward()
    assert torch.equal(x.grad, reference_x.grad)
    assert layer.weight.grad is None
    assert layer.bias.grad is None


def test_peft_adapters_are_not_float8() -> None:
    model, _, _ = _fp8_model()
    prepare_fp8_cast_linears_for_lora(model)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=["to_q"], init_lora_weights=True))
    lora_params = {name: p for name, p in peft_model.named_parameters() if "lora_" in name}
    assert lora_params
    assert all(p.dtype == torch.float32 for p in lora_params.values())

    peft_model(torch.randn(ROWS, IN_FEATURES, dtype=torch.bfloat16)).float().sum().backward()
    assert all(p.grad is not None for p in lora_params.values())
    assert all(torch.isfinite(p.grad).all() for p in lora_params.values())


def test_quantize_model_rejects_fp8_cast() -> None:
    with pytest.raises(ValueError, match="load_transformer"):
        quantize_model(nn.Linear(4, 4), precision="fp8-cast")


def test_with_quantization_policy_chains_ops() -> None:
    base = SDOps("base").with_matching(prefix="model.")
    extra = SDOps("extra").with_kv_operation(key_suffix=".weight", operation=lambda _key, _value: [])
    op = ModuleOps(name="retype", matcher=lambda _model: True, mutator=lambda model: model)

    assert with_quantization_policy(base, (), None) == (base, ())
    sd_ops, module_ops = with_quantization_policy(base, (), QuantizationPolicy(sd_ops=extra, module_ops=(op,)))
    assert sd_ops.mapping == (*base.mapping, *extra.mapping)
    assert module_ops == (op,)
    sd_ops, module_ops = with_quantization_policy(base, (op,), QuantizationPolicy(sd_ops=None, module_ops=(op,)))
    assert sd_ops is base
    assert module_ops == (op, op)
