"""fp8-gemm base for LoRA training: the per-row e4m3 quantizer, Fp8RowwiseLinear layers whose float8 GEMMs track the
bf16 reference and pass gradients to their inputs only, the upcast fallback for short inputs and for devices without
fp8 GEMMs, fp32 PEFT adapters, Dynamo tracing of forward and backward, and the loader-only contract."""

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from pydantic import ValidationError
from safetensors.torch import save_file
from torch import nn

from ltx_core.quantization.fp8_cast import E4M3_MAX, Fp8RowwiseLinear, supports_fp8_gemm
from ltx_trainer.config import AccelerationConfig
from ltx_trainer.quantization import (
    fp8_gemm_policy,
    prepare_fp8_gemm_linears_for_lora,
    quantize_model,
    quantize_rows_e4m3,
)

IN_FEATURES, OUT_FEATURES, MIN_ROWS = 64, 32, 16
HAS_FP8_GEMM = torch.cuda.is_available() and supports_fp8_gemm(torch.device("cuda"))
needs_fp8_gemm = pytest.mark.skipif(not HAS_FP8_GEMM, reason="needs a GPU with float8 GEMMs (compute capability 8.9+)")


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


def _fp8_gemm_model(tmp_path: Path, device: str) -> tuple[_Model, torch.Tensor, torch.Tensor]:
    """A model prepared the way the trainer prepares an fp8-gemm base, plus the bf16 weight and bias its float8
    values round to. The policy's module ops are applied directly: their matchers look for an ``LTXModel``."""
    torch.manual_seed(0)
    model = _Model().to(device)
    layer = model.transformer_blocks[0].attn1.to_q
    weight8 = layer.weight.detach().to(torch.float8_e4m3fn)
    bias8 = layer.bias.detach().to(torch.float8_e4m3fn)
    checkpoint = tmp_path / "transformer.safetensors"
    save_file({"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": layer.weight.detach().cpu()}, checkpoint)
    for op in fp8_gemm_policy(checkpoint, min_rows=MIN_ROWS).module_ops:
        op.mutator(model)
    layer.weight.data, layer.bias.data = weight8, bias8
    model.requires_grad_(False)
    assert prepare_fp8_gemm_linears_for_lora(model) == 1
    return model, weight8.to(torch.bfloat16), bias8.to(torch.bfloat16)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def test_quantize_rows_e4m3_scales_each_row_to_the_e4m3_range() -> None:
    torch.manual_seed(1)
    x = torch.randn(6, IN_FEATURES, dtype=torch.bfloat16) * torch.tensor([[1e-3], [1.0], [40.0], [0.0], [7.0], [2.0]])
    x8, scale = quantize_rows_e4m3(x)

    assert x8.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (6, 1)
    row_max = x8.float().abs().amax(dim=-1)
    assert torch.equal(row_max[[0, 1, 2, 4, 5]], torch.full((5,), E4M3_MAX))
    assert row_max[3] == 0
    assert scale[3] == 1
    assert _relative_error(x8.float() * scale, x) < 0.05


@needs_fp8_gemm
def test_fp8_gemm_linear_tracks_reference_and_grads_only_inputs(tmp_path: Path) -> None:
    model, weight, bias = _fp8_gemm_model(tmp_path, "cuda")
    layer = model.transformer_blocks[0].attn1.to_q
    assert isinstance(layer, Fp8RowwiseLinear)
    assert type(layer).__name__ == "Fp8RowwiseLinearTrain"
    assert layer.compute_dtype == torch.bfloat16
    assert layer.weight.dtype == torch.float8_e4m3fn

    x = torch.randn(2, MIN_ROWS, IN_FEATURES, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out = model(x)
    reference_x = x.detach().requires_grad_()
    reference = nn.functional.linear(reference_x, weight, bias)
    assert out.dtype == torch.bfloat16
    assert out.shape == reference.shape
    assert _relative_error(out, reference) < 0.05

    out.float().square().sum().backward()
    reference.float().square().sum().backward()
    assert _relative_error(x.grad, reference_x.grad) < 0.05
    assert layer.weight.grad is None
    assert layer.bias.grad is None


@pytest.mark.parametrize(
    ("device", "rows"),
    [
        pytest.param("cpu", 2 * MIN_ROWS, id="device-without-fp8-gemms"),
        pytest.param("cuda", MIN_ROWS - 1, id="below-min-rows", marks=needs_fp8_gemm),
    ],
)
def test_other_inputs_take_the_exact_upcast_path(tmp_path: Path, device: str, rows: int) -> None:
    model, weight, bias = _fp8_gemm_model(tmp_path, device)
    x = torch.randn(rows, IN_FEATURES, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = model(x)
    reference_x = x.detach().requires_grad_()
    reference = nn.functional.linear(reference_x, weight, bias)
    assert torch.equal(out, reference)
    out.float().square().sum().backward()
    reference.float().square().sum().backward()
    assert torch.equal(x.grad, reference_x.grad)


@needs_fp8_gemm
def test_peft_adapters_are_fp32(tmp_path: Path) -> None:
    model, _, _ = _fp8_gemm_model(tmp_path, "cuda")
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=["to_q"], init_lora_weights=True))
    lora_params = {name: p for name, p in peft_model.named_parameters() if "lora_" in name}
    assert lora_params
    assert all(p.dtype == torch.float32 for p in lora_params.values())

    peft_model(torch.randn(MIN_ROWS, IN_FEATURES, dtype=torch.bfloat16, device="cuda")).float().sum().backward()
    assert all(p.grad is not None for p in lora_params.values())
    assert all(torch.isfinite(p.grad).all() for p in lora_params.values())


@needs_fp8_gemm
def test_forward_and_backward_trace_under_dynamo(tmp_path: Path) -> None:
    model, _, _ = _fp8_gemm_model(tmp_path, "cuda")
    x = torch.randn(MIN_ROWS, IN_FEATURES, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    eager = model(x)
    eager.float().square().sum().backward()
    eager_grad, x.grad = x.grad, None

    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    out = compiled(x)
    out.float().square().sum().backward()
    assert torch.equal(out, eager)
    assert torch.equal(x.grad, eager_grad)


@needs_fp8_gemm
@pytest.mark.parametrize("rows", [MIN_ROWS, MIN_ROWS - 1], ids=["fp8-gemm-path", "upcast-path"])
def test_fp32_inputs_under_autocast_are_cast_before_the_weight(tmp_path: Path, rows: int) -> None:
    model, weight, bias = _fp8_gemm_model(tmp_path, "cuda")
    x = torch.randn(rows, IN_FEATURES, dtype=torch.float32, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    assert out.dtype == torch.bfloat16
    reference = nn.functional.linear(x.detach().to(torch.bfloat16), weight, bias)
    assert _relative_error(out, reference) < 0.05
    out.float().square().sum().backward()
    assert x.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all()


def test_quantize_model_rejects_fp8_gemm() -> None:
    with pytest.raises(ValueError, match="fp8_gemm_policy"):
        quantize_model(nn.Linear(4, 4), precision="fp8-gemm")


def test_fp8_gemm_min_rows_is_positive() -> None:
    assert AccelerationConfig(fp8_gemm_min_rows=1).fp8_gemm_min_rows == 1
    with pytest.raises(ValidationError):
        AccelerationConfig(fp8_gemm_min_rows=0)
