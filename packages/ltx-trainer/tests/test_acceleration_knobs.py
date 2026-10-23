"""The step-time and startup knobs: metrics read-back cadence, compiled block forwards that leave the module
tree alone, quantized blocks that stay on their device, and the config contract around them."""

import pytest
import torch
from optimum.quanto.nn import QLinear
from pydantic import ValidationError
from torch import nn

from ltx_trainer.config import AccelerationConfig
from ltx_trainer.quantization import quantize_model
from ltx_trainer.trainer import compile_block_forwards, metrics_due

TOTAL_STEPS = 100


@pytest.mark.parametrize(
    ("step", "interval", "expected"),
    [
        (7, 1, True),  # interval 1: every step
        (7, 50, False),
        (50, 50, True),
        (100, 50, True),  # last step always
        (99, 50, False),
        (100, 7, True),  # last step even when the interval does not divide it
    ],
)
def test_metrics_due(step: int, interval: int, expected: bool) -> None:
    assert metrics_due(step, interval, TOTAL_STEPS) is expected


def test_metrics_sync_interval_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AccelerationConfig(metrics_sync_interval=0)


def test_compile_block_forwards_keeps_module_tree() -> None:
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
    x = torch.randn(3, 8)
    before = [block(x) for block in blocks]
    keys = list(blocks.state_dict())

    assert compile_block_forwards(blocks, backend="eager") == 2

    assert list(blocks.state_dict()) == keys  # no wrapper module, no renamed parameters
    for block, expected in zip(blocks, before, strict=True):
        assert "forward" in vars(block)  # the bound method was replaced on the instance
        assert torch.equal(block(x), expected)


def test_quantize_model_keep_on_device() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # quanto swaps a module's children, as it does the Linears inside each real transformer block
            self.transformer_blocks = nn.ModuleList([nn.Sequential(nn.Linear(16, 16)) for _ in range(2)])
            self.proj_out = nn.Linear(16, 4)  # excluded from quantization by name

    model = quantize_model(_Model(), precision="int8-quanto", device="cpu", keep_on_device=True)
    assert all(isinstance(block[0], QLinear) for block in model.transformer_blocks)
    assert not isinstance(model.proj_out, QLinear)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert torch.isfinite(model.transformer_blocks[0](torch.randn(2, 16))).all()
