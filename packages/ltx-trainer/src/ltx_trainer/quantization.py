# Adapted from: https://github.com/bghira/SimpleTuner
# With improvements from: https://github.com/ostris/ai-toolkit
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from torch.autograd.function import FunctionCtx

from ltx_trainer import logger

if TYPE_CHECKING:
    from ltx_core.quantization import QuantizationPolicy

QuantizationOptions = Literal[
    "fp8-cast",
    "fp8-gemm",
    "int8-cast",
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "fp8-quanto",
    "fp8uz-quanto",
]

# Modules to exclude from quantization.
# These are glob patterns passed to quanto's `exclude` parameter.
# When quantizing the full model at once, these patterns match against full module paths.
# When quantizing block-by-block, we also use SKIP_ROOT_MODULES for top-level modules.
EXCLUDE_PATTERNS = [
    # Input/output projection layers
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    # Timestep embedding layers - int4 tinygemm requires strict bfloat16 input
    # and these receive float32 sinusoidal embeddings that are cast to bfloat16
    "*adaln*",
    "time_proj",
    "timestep_embedder*",
    # Caption/text projection layers
    "caption_projection*",
    "audio_caption_projection*",
    # Normalization layers (usually excluded from quantization)
    "*norm*",
]

# Top-level modules to skip entirely during block-by-block quantization.
# These are exact matches against model.named_children() names.
# (Needed because quanto's exclude patterns don't work when calling quantize() directly on a module)
SKIP_ROOT_MODULES = {
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    "audio_caption_projection",
}


def fp8_cast_policy(checkpoint_path: str | Path) -> "QuantizationPolicy":
    """ltx-core's fp8-cast policy for a transformer checkpoint.
    Applied by ``load_transformer(quantization=...)``: the policy's state-dict ops write the transformer-block
    Linears (``to_q/k/v``, ``to_out.0``, ``ff.net.*`` and their audio twins) as ``float8_e4m3fn`` while the
    checkpoint streams in, and its module ops retype those layers to ``Fp8CastLinear``, which upcasts the weight
    to the activation dtype inside a class-level forward. The same storage inference uses; no quantization pass.
    Unlike the quanto options this is a plain ``nn.Linear`` subclass: no tensor-subclass dispatch, autograd
    flows through ``F.linear``, and ``torch.compile`` can trace it.
    """
    from ltx_core.quantization.fp8_cast import build_policy  # noqa: PLC0415

    return build_policy(str(checkpoint_path))


def fp8_gemm_policy(checkpoint_path: str | Path, min_rows: int = 1024) -> "QuantizationPolicy":
    """:func:`fp8_cast_policy` with the block Linears' long inputs multiplied in float8: ltx-core's
    ``with_rowwise_fp8_gemm`` retypes them to ``Fp8RowwiseLinear``, the rowwise fp8 GEMM the pipelines use for
    inference. A bf16 activation of at least ``min_rows`` rows is rounded to e4m3 with one scale per row and
    multiplied with the stored e4m3 weight through ``torch._scaled_mm``; shorter inputs upcast like fp8-cast.
    Weights and their loading are those of :func:`fp8_cast_policy`.
    """
    from ltx_core.quantization.fp8_cast import with_rowwise_fp8_gemm  # noqa: PLC0415

    return with_rowwise_fp8_gemm(fp8_cast_policy(checkpoint_path), min_rows=min_rows)


def int8_cast_policy(checkpoint_path: str | Path | None = None) -> "QuantizationPolicy":  # noqa: ARG001
    """ltx-core's int8-cast policy: the transformer-block Linears are stored as int8 with one bf16 scale per output
    row, written while the checkpoint streams in, and dequantized inside a class-level forward
    (``Int8CastLinear``). The recipe is optimum-quanto's absmax int8, so the weights are what ``int8-quanto`` trains
    against, bit for bit, without the tensor subclass. ``checkpoint_path`` is accepted for a uniform signature with
    :func:`fp8_cast_policy`; the bf16 checkpoint needs nothing read ahead of time.
    """
    from ltx_core.quantization.int8_cast import build_policy  # noqa: PLC0415

    return build_policy()


class _Int8FrozenLinearFunction(torch.autograd.Function):
    """Linear over a frozen per-row int8 weight: dequantize for the forward GEMM, save only the int8 weight and its
    scale, dequantize again for the input gradient. No weight or bias gradient: the base model is frozen."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, x: torch.Tensor, weight8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        ctx.save_for_backward(weight8, scale)
        bias = bias.to(x.dtype) if bias is not None else None
        return torch.nn.functional.linear(x, weight8.to(x.dtype) * scale.to(x.dtype)[:, None], bias)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        weight8, scale = ctx.saved_tensors
        return grad_out.matmul(weight8.to(grad_out.dtype) * scale.to(grad_out.dtype)[:, None]), None, None, None


def _int8_train_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _Int8FrozenLinearFunction.apply(x, self.weight, self.weight_scale, self.bias)


def prepare_int8_cast_linears_for_lora(model: torch.nn.Module) -> int:
    """Make a model loaded with :func:`int8_cast_policy` ready for LoRA training. Returns the layer count.
    Same two steps as :func:`prepare_fp8_cast_linears_for_lora`: a ``compute_dtype`` so PEFT creates bf16 (then
    fp32) adapters rather than int8 ones, and a class-level forward whose backward saves the int8 weight and scale
    instead of the dequantized bf16 copy.
    """
    from ltx_core.quantization.int8_cast import Int8CastLinear  # noqa: PLC0415

    train_cls = type("Int8CastLinearTrain", (Int8CastLinear,), {"forward": _int8_train_forward})
    count = 0
    for module in model.modules():
        if type(module) is Int8CastLinear:
            module.__class__ = train_cls
            module.compute_dtype = torch.bfloat16
            count += 1
    return count


class _Fp8FrozenLinearFunction(torch.autograd.Function):
    """Linear over a frozen float8 weight.
    ``F.linear(x, weight8.to(x.dtype))`` would save the upcast bf16 weight for backward, which without gradient
    checkpointing keeps a second bf16 copy of every quantized Linear alive. This saves the float8 parameter
    instead and upcasts again for the input gradient. No weight or bias gradient: the base model is frozen.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, weight8: torch.Tensor, bias8: torch.Tensor | None) -> torch.Tensor:
        ctx.save_for_backward(weight8)
        bias = bias8.to(x.dtype) if bias8 is not None else None
        return torch.nn.functional.linear(x, weight8.to(x.dtype), bias)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (weight8,) = ctx.saved_tensors
        return grad_out.matmul(weight8.to(grad_out.dtype)), None, None


def _fp8_train_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _Fp8FrozenLinearFunction.apply(x, self.weight, self.bias)


def prepare_fp8_cast_linears_for_lora(model: torch.nn.Module) -> int:
    """Make a model loaded with :func:`fp8_cast_policy` ready for LoRA training. Returns the layer count.
    * PEFT creates a new adapter in the base layer's weight dtype when that dtype is floating point, which
      would make the adapters float8. PEFT prefers a ``compute_dtype`` attribute when present, so the adapters
      are created in bfloat16 (and then upcast to float32 by PEFT's default ``autocast_adapter_dtype``).
    * The forward is swapped at class level (keeping ``torch.compile`` happy) for one whose backward saves
      the float8 weight rather than its bf16 upcast.
    """
    from ltx_core.quantization.fp8_cast import Fp8CastLinear  # noqa: PLC0415

    train_cls = type("Fp8CastLinearTrain", (Fp8CastLinear,), {"forward": _fp8_train_forward})
    count = 0
    for module in model.modules():
        if type(module) is Fp8CastLinear:
            module.__class__ = train_cls
            module.compute_dtype = torch.bfloat16
            count += 1
    return count


def quantize_rows_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round the rows of a 2-D tensor to e4m3 with one scale per row: ``(x8, scale)`` with ``x8 * scale``
    approximating ``x``. Each row's absolute maximum maps to the e4m3 maximum; an all-zero row keeps scale 1.
    ``scale`` is an fp32 ``[rows, 1]`` column. Plain tensor ops, so a compile backend fuses the cast into the
    kernel producing ``x``.
    """
    from ltx_core.quantization.fp8_cast import E4M3_MAX  # noqa: PLC0415

    amax = x.abs().amax(dim=-1, keepdim=True).float()
    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def _scaled_rows_mm(
    a8: torch.Tensor, a_scale: torch.Tensor, b_col_major: torch.Tensor, unit: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``(a8 @ b) * a_scale`` in float8: ``torch._scaled_mm`` with unit scales (the cuBLASLt fp8 GEMM, fp32
    accumulation, bf16 product), the row scales applied to the product. ``b_col_major`` is the transposed view of
    a row-major e4m3 matrix, the layout the GEMM's second operand needs. ``unit`` is the fp32 scalar 1 the layer
    keeps: it has to reach the GEMM as a graph input, not as a tensor created here (see the Function below)."""
    out = torch._scaled_mm(a8, b_col_major, scale_a=unit, scale_b=unit, out_dtype=torch.bfloat16, use_fast_accum=False)
    return (out.float() * a_scale).to(out_dtype)


class _Fp8GemmFrozenLinearFunction(torch.autograd.Function):
    """Float8 GEMM over a frozen e4m3 weight for a 2-D activation already rounded per row by the caller.
    The rounding stays outside this Function on purpose: Dynamo traces an autograd Function as a subgraph and
    cannot lift the rounding's intermediates into it, while plain tensor ops in the module forward are traced and
    fused by Inductor. ``x2`` is the bf16 activation the gradient flows back to; ``x8`` and ``scale`` are its e4m3
    rows and per-row scales. Backward rounds the output gradient per row the same way and multiplies it with a
    transposed copy of the weight: the fake-tensor kernel for ``_scaled_mm`` under tracing insists on a column-major
    second operand, which a plain ``.t()`` of the stored weight is not. ``unit`` is the layer's fp32 scalar 1,
    passed in and saved for backward so that both GEMMs see it as a graph input: a scalar created inside the
    traced forward or backward is a graph constant, and Inductor then compiles the GEMM through a slower path
    (measured: +14% on an audiovisual step). No weight or bias gradient: the base is frozen.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x2: torch.Tensor,  # noqa: ARG004
        x8: torch.Tensor,
        scale: torch.Tensor,
        weight8: torch.Tensor,
        unit: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight8, unit)
        return _scaled_rows_mm(x8, scale, weight8.t(), unit, torch.bfloat16)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        weight8, unit = ctx.saved_tensors
        g8, scale = quantize_rows_e4m3(grad_out)
        return _scaled_rows_mm(g8, scale, weight8.t().contiguous().t(), unit, grad_out.dtype), None, None, None, None


def _fp8_gemm_train_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    from ltx_core.quantization.fp8_cast import supports_fp8_gemm  # noqa: PLC0415

    in_features = x.shape[-1]
    rows = x.numel() // in_features
    fp8_gemm = rows >= self._min_rows and supports_fp8_gemm(x.device) and not self._with_stochastic_rounding
    if fp8_gemm and x.dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
        # Norm outputs arrive in fp32 under autocast; F.linear would cast them to the autocast dtype anyway. Only
        # for the float8 path: on the upcast path an explicit cast is not free (about 3 GB of extra activations on
        # an audio-only run without gradient checkpointing, for no step-time gain).
        x = x.to(torch.get_autocast_dtype("cuda"))
    if not fp8_gemm or x.dtype != torch.bfloat16:
        return _Fp8FrozenLinearFunction.apply(x, self.weight, self.bias)
    x2 = x.reshape(rows, in_features)
    x8, row_scale = quantize_rows_e4m3(x2)
    out = _Fp8GemmFrozenLinearFunction.apply(x2, x8, row_scale, self.weight, self.fp8_gemm_unit)
    out = out.view(*x.shape[:-1], self.out_features)
    return out if self.bias is None else out + self.bias.to(out.dtype)


def prepare_fp8_gemm_linears_for_lora(model: torch.nn.Module) -> int:
    """Make a model loaded with :func:`fp8_gemm_policy` ready for LoRA training. Returns the layer count.
    Same two steps as :func:`prepare_fp8_cast_linears_for_lora`, applied to the ``Fp8RowwiseLinear`` layers: a
    ``compute_dtype`` so PEFT creates bf16 (then fp32) adapters, and a class-level forward that keeps the rowwise
    float8 GEMM for long inputs and adds a backward that computes the input gradient the same way. Inputs below the
    policy's row threshold, inputs that are not bf16, and devices without fp8 GEMMs take the fp8-cast training path.
    On the float8 path an fp32 input under autocast is cast to the autocast dtype first. Each layer also gets the
    fp32 scalar 1 its GEMMs use as the unit scale (``fp8_gemm_unit``), kept as a plain attribute so it stays out
    of the state dict.
    """
    from ltx_core.quantization.fp8_cast import Fp8RowwiseLinear  # noqa: PLC0415

    train_cls = type("Fp8RowwiseLinearTrain", (Fp8RowwiseLinear,), {"forward": _fp8_gemm_train_forward})
    count = 0
    for module in model.modules():
        if type(module) is Fp8RowwiseLinear:
            module.__class__ = train_cls
            module.compute_dtype = torch.bfloat16
            module.fp8_gemm_unit = torch.ones((), device=module.weight.device, dtype=torch.float32)
            count += 1
    return count


def quantize_model(
    model: torch.nn.Module,
    precision: QuantizationOptions,
    quantize_activations: bool = False,
    device: torch.device | str | None = None,
    keep_on_device: bool = False,
) -> torch.nn.Module:
    """
    Quantize a model using optimum-quanto.
    For large models with transformer_blocks, this function quantizes block-by-block
    on GPU then moves back to CPU, which is much faster than quantizing on CPU and
    uses less peak VRAM than loading the entire model to GPU at once.
    Args:
        model: The model to quantize.
        precision: The quantization precision (e.g. "int8-quanto", "fp8-quanto").
        quantize_activations: Whether to quantize activations in addition to weights.
        device: Device to use for quantization. If None, uses CUDA if available, else CPU.
        keep_on_device: Leave the quantized model on ``device`` instead of returning it to the device it
            came from. Saves two transfers of the model when it is about to be trained on ``device`` anyway.
    Returns:
        The quantized model.
    """
    if precision in ("fp8-cast", "fp8-gemm", "int8-cast"):
        raise ValueError(
            f"{precision} is applied while the checkpoint loads: pass {precision.replace('-', '_')}_policy(...) to "
            "load_transformer() instead of calling quantize_model()."
        )

    from optimum.quanto import freeze, quantize  # noqa: PLC0415

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    weight_quant = _get_quanto_dtype(precision)

    if quantize_activations:
        logger.debug("Quantizing model weights and activations")
        activations_quant = weight_quant
    else:
        activations_quant = None

    # Remember original device to restore after quantization
    original_device = next(model.parameters()).device

    # Check if model has transformer_blocks for block-by-block quantization
    if hasattr(model, "transformer_blocks"):
        logger.debug("Quantizing model using block-by-block approach for memory efficiency")
        _quantize_blockwise(
            model,
            weight_quant=weight_quant,
            activations_quant=activations_quant,
            device=device,
            keep_on_device=keep_on_device,
        )
    else:
        # Fallback: quantize entire model at once
        model.to(device)
        quantize(model, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(model)

    # Restore model to original device
    if not keep_on_device:
        model.to(original_device)

    return model


def _quantize_blockwise(
    model: torch.nn.Module,
    weight_quant: torch.dtype,
    activations_quant: torch.dtype | None,
    device: torch.device,
    keep_on_device: bool = False,
) -> None:
    """Quantize a model block-by-block using optimum-quanto.
    This approach:
    1. Moves each transformer block to GPU
    2. Quantizes on GPU (fast!)
    3. Freezes the quantized weights
    4. Moves back to CPU
    This is much faster than quantizing on CPU and uses less peak VRAM
    than loading the entire model to GPU.
    """
    from optimum.quanto import freeze, quantize  # noqa: PLC0415

    original_dtype = next(model.parameters()).dtype
    transformer_blocks = list(model.transformer_blocks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Quantizing transformer blocks", total=len(transformer_blocks))

        for block in transformer_blocks:
            # Move block to GPU
            block.to(device, dtype=original_dtype, non_blocking=True)

            # Quantize on GPU
            quantize(block, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
            freeze(block)

            # Move back to CPU to free up VRAM for next block
            if not keep_on_device:
                block.to("cpu", non_blocking=True)

            progress.advance(task)

    # Quantize remaining non-transformer-block modules (e.g., embeddings, timestep projections)
    # Skip modules that should not be quantized (patchify_proj, proj_out, etc.)
    logger.debug("Quantizing remaining model components")

    for name, module in model.named_children():
        if name == "transformer_blocks":
            continue  # Already quantized

        if name in SKIP_ROOT_MODULES:
            logger.debug(f"Skipping quantization for module: {name}")
            continue  # Don't quantize these modules

        # Move to device, quantize, freeze, move back
        module.to(device, dtype=original_dtype, non_blocking=True)
        quantize(module, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(module)
        if not keep_on_device:
            module.to("cpu", non_blocking=True)


def _get_quanto_dtype(precision: QuantizationOptions) -> torch.dtype:
    """Map precision string to quanto dtype."""
    from optimum.quanto import (  # noqa: PLC0415
        qfloat8,
        qfloat8_e4m3fnuz,
        qint2,
        qint4,
        qint8,
    )

    if precision == "int2-quanto":
        return qint2
    elif precision == "int4-quanto":
        return qint4
    elif precision == "int8-quanto":
        return qint8
    elif precision in ("fp8-quanto", "fp8uz-quanto"):
        if torch.backends.mps.is_available():
            raise ValueError("FP8 quantization is not supported on MPS devices. Use int2, int4, or int8 instead.")
        if precision == "fp8-quanto":
            return qfloat8
        elif precision == "fp8uz-quanto":
            return qfloat8_e4m3fnuz

    raise ValueError(f"Invalid quantization precision: {precision}")
