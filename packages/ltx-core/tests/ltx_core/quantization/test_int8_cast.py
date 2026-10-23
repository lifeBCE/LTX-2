"""The int8-cast policy: load-time quantization emits int8 data plus per-row scales, the retyped layer computes the
dequantized linear, and the fuse rule folds a LoRA delta while keeping the scale companion in sync."""

import torch
from torch import nn

from ltx_core.loader.primitives import StateDict
from ltx_core.quantization.int8_cast import (
    RETYPE_INT8_CAST,
    TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP,
    Int8CastLinear,
    dequantize_int8_per_row,
    int8_cast_fuse_rule,
    quantize_weight_int8_per_row,
)

OUT, IN, ROWS = 32, 64, 5
KEY = "transformer_blocks.0.attn1.to_q.weight"


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(IN, OUT, dtype=torch.bfloat16)
        self.to_gate_logits = nn.Linear(IN, OUT, dtype=torch.bfloat16)  # outside the cast set


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _Attention()


class _Model(nn.Module):
    """``transformer_blocks.N.attn1.to_q`` naming, as the policy's name matcher expects."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])


def test_quantize_round_trip_bounds() -> None:
    torch.manual_seed(0)
    weight = torch.randn(OUT, IN, dtype=torch.bfloat16) * 0.02
    weight[3] = 0  # an all-zero row must not divide by zero
    data, scale = quantize_weight_int8_per_row(weight)
    assert data.dtype == torch.int8
    assert data.shape == (OUT, IN)
    assert scale.dtype == torch.bfloat16
    assert scale.shape == (OUT,)
    assert data.abs().max() <= 127
    back = dequantize_int8_per_row(data, scale, torch.bfloat16)
    assert torch.isfinite(back).all()
    assert torch.equal(back[3], torch.zeros(IN, dtype=torch.bfloat16))
    assert ((back.float() - weight.float()).abs().max() / weight.abs().max()) < 1 / 127


def test_downcast_sd_ops_emit_weight_and_scale() -> None:
    torch.manual_seed(0)
    weight = torch.randn(OUT, IN, dtype=torch.bfloat16)
    results = TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP.apply_to_key_value(KEY, weight)
    assert [r.new_key for r in results] == [KEY, "transformer_blocks.0.attn1.to_q.weight_scale"]
    data, scale = quantize_weight_int8_per_row(weight)
    assert torch.equal(results[0].new_value, data)
    assert torch.equal(results[1].new_value, scale)
    # bias and non-block keys pass through untouched
    bias = torch.randn(OUT, dtype=torch.bfloat16)
    (passthrough,) = TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP.apply_to_key_value(
        "transformer_blocks.0.attn1.to_q.bias", bias
    )
    assert passthrough.new_key.endswith(".bias")
    assert torch.equal(passthrough.new_value, bias)
    (other,) = TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP.apply_to_key_value("patchify_proj.weight", weight)
    assert other.new_value.dtype == torch.bfloat16


def test_retyped_layer_computes_dequantized_linear() -> None:
    torch.manual_seed(0)
    model = _Model()
    layer = model.transformer_blocks[0].attn1.to_q
    weight, bias = layer.weight.detach().clone(), layer.bias.detach().clone()
    RETYPE_INT8_CAST.mutator(model)
    assert isinstance(layer, Int8CastLinear)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (OUT,)
    assert not isinstance(model.transformer_blocks[0].attn1.to_gate_logits, Int8CastLinear)
    # what the loader does: assign the sd-op outputs onto the retyped parameters
    sd = {
        r.new_key.removeprefix("transformer_blocks.0.attn1.to_q."): r.new_value
        for r in TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP.apply_to_key_value(KEY, weight)
    }
    sd["bias"] = bias
    layer.load_state_dict(sd, assign=True)
    x = torch.randn(ROWS, IN, dtype=torch.bfloat16)
    expected = nn.functional.linear(x, dequantize_int8_per_row(sd["weight"], sd["weight_scale"], torch.bfloat16), bias)
    assert torch.equal(layer(x), expected)


def test_fuse_rule_requantizes_and_updates_scale() -> None:
    torch.manual_seed(0)
    weight = torch.randn(OUT, IN, dtype=torch.bfloat16)
    data, scale = quantize_weight_int8_per_row(weight)
    delta = torch.randn(OUT, IN, dtype=torch.bfloat16) * 0.5
    sd = {KEY: data, KEY.removesuffix(".weight") + ".weight_scale": scale}
    model_sd = StateDict(
        sd=sd, device=torch.device("cpu"), size=sum(v.nbytes for v in sd.values()), dtype={torch.int8, torch.bfloat16}
    )
    fused = int8_cast_fuse_rule.fuse_fn(KEY, data, delta.clone(), model_sd)
    assert set(fused) == {KEY, "transformer_blocks.0.attn1.to_q.weight_scale"}
    assert fused[KEY].dtype == torch.int8
    target = dequantize_int8_per_row(data, scale, torch.bfloat16) + delta
    back = dequantize_int8_per_row(fused[KEY], fused["transformer_blocks.0.attn1.to_q.weight_scale"], torch.bfloat16)
    assert ((back.float() - target.float()).abs().max() / target.abs().max()) < 1 / 127
    # a bf16 key (no scale companion) takes the plain bf16 fuse
    plain = int8_cast_fuse_rule.fuse_fn("patchify_proj.weight", weight, delta.clone(), model_sd)
    assert set(plain) == {"patchify_proj.weight"}
    assert torch.equal(plain["patchify_proj.weight"], (delta + weight))
