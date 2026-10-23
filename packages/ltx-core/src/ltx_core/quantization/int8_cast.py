"""Int8-cast: per-row symmetric int8 storage for the transformer-block Linears, dequantized in forward.

Same layer set and load-time mechanics as :mod:`ltx_core.quantization.fp8_cast`: state-dict ops quantize the
checkpoint's bf16 weights while they stream in, module ops retype the layers on the meta model so the parameters
match what the loader assigns. The recipe is optimum-quanto's symmetric absmax int8 (one scale per output row,
``amax / 127`` in the weight dtype, round to nearest), so the stored weights equal what ``int8-quanto`` computes,
bit for bit, held in a plain ``nn.Linear`` with no tensor subclass. Weights keep about five times more precision
than fp8 e4m3 at the same byte per weight.
"""

import torch

from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer.model import LTXModel
from ltx_core.quantization.fp8_cast import _FP8_CAST_KEY_PREFIX, _FP8_CAST_LINEAR_SUFFIXES, _is_fp8_cast_linear
from ltx_core.quantization.policy import QuantizationPolicy

INT8_MIN = torch.iinfo(torch.int8).min
INT8_MAX = torch.iinfo(torch.int8).max
SCALE_SUFFIX = ".weight_scale"


def quantize_weight_int8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to int8 with one symmetric absmax scale per output row.
    The scale is ``amax / 127`` computed in the weight's own dtype and the data is ``round(weight / scale)``,
    which is optimum-quanto's ``qint8`` recipe along axis 0. Returns ``(data int8 [out, in], scale [out])`` with
    the scale in the weight dtype. An all-zero row keeps a scale of 1 instead of dividing by zero.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}")
    scale = weight.abs().amax(dim=1, keepdim=True) / INT8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    data = torch.clamp(torch.round(weight / scale), INT8_MIN, INT8_MAX).to(torch.int8)
    return data, scale.squeeze(1)


def dequantize_int8_per_row(data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``data * scale`` per output row, in *dtype*."""
    return data.to(dtype) * scale.to(dtype)[:, None]


class Int8CastLinear(torch.nn.Linear):
    """``nn.Linear`` whose ``weight`` is int8 with a per-row ``weight_scale``, dequantized in a class-level forward.
    Used via ``__class__`` reassignment (not subclassing) so the loader's parameters stay in place. The forward is
    class-level so ``torch.compile`` traces it; the dequantization is two elementwise ops that Inductor fuses into
    one kernel ahead of the GEMM. The bias stays in the checkpoint dtype.
    """

    weight_scale: torch.nn.Parameter

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        weight = dequantize_int8_per_row(self.weight, self.weight_scale, input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(input, weight, bias)


def _retype_to_int8_cast(layer: torch.nn.Linear) -> None:
    """Retype *layer* in place: int8 ``weight`` and a per-row ``weight_scale`` the loader fills from the sd ops."""
    layer.__class__ = Int8CastLinear
    layer.weight = torch.nn.Parameter(
        torch.empty(layer.weight.shape, dtype=torch.int8, device=layer.weight.device), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty(layer.weight.shape[0], dtype=torch.bfloat16, device=layer.weight.device), requires_grad=False
    )


def _retype_block_linears(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and _is_fp8_cast_linear(name):
            _retype_to_int8_cast(module)
    return model


def _int8_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    data, scale = quantize_weight_int8_per_row(value)
    return [
        KeyValueOperationResult(key, data),
        KeyValueOperationResult(key.removesuffix(".weight") + SCALE_SUFFIX, scale),
    ]


def _build_downcast_map() -> SDOps:
    ops = SDOps("TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP")
    for suffix in _FP8_CAST_LINEAR_SUFFIXES:
        ops = ops.with_kv_operation(
            key_prefix=_FP8_CAST_KEY_PREFIX, key_suffix=suffix + ".weight", operation=_int8_downcast
        )
    return ops


TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP = _build_downcast_map()

RETYPE_INT8_CAST = ModuleOps(
    name="int8_cast_block_linears",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=_retype_block_linears,
)


def _int8_cast_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequantize, add the bf16 LoRA delta, requantize with a fresh per-row scale; keep ``.weight_scale`` in sync.
    Layers outside the int8 set (no companion scale) take the plain bf16 fuse.
    """
    scale_key = key.removesuffix(".weight") + SCALE_SUFFIX
    if weight.dtype != torch.int8 or scale_key not in model_sd.sd:
        return bf16_fuse_rule(key, weight, deltas, model_sd)
    scale = model_sd.sd[scale_key]
    deltas.add_(dequantize_int8_per_row(weight, scale, deltas.dtype))
    data, new_scale = quantize_weight_int8_per_row(deltas)
    return {key: data, scale_key: new_scale.to(scale.dtype)}


int8_cast_fuse_rule = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_int8_cast_fuse)


def build_policy() -> QuantizationPolicy:
    """Int8-cast for a bf16 checkpoint: int8 block Linears written at load time, dequantized in forward."""
    return QuantizationPolicy(
        sd_ops=TRANSFORMER_LINEAR_INT8_DOWNCAST_MAP,
        module_ops=(RETYPE_INT8_CAST,),
        fuse_rule=int8_cast_fuse_rule,
    )
