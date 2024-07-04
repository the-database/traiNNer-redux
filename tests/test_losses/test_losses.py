from collections.abc import Callable
from typing import TypedDict

import pytest
import torch
from torch import Tensor, nn
from traiNNer.losses.adists_loss import ADISTSLoss
from traiNNer.losses.basic_loss import (
    CharbonnierLoss,
    ColorLoss,
    HSLuvLoss,
    L1Loss,
    LumaLoss,
    MSELoss,
)
from traiNNer.losses.dists_loss import DISTSLoss
from traiNNer.losses.mssim_loss import MSSIMLoss
from traiNNer.losses.perceptual_loss import PerceptualLoss

LOSS_FUNCTIONS = [
    L1Loss(),
    CharbonnierLoss(),
    MSELoss(),
    MSSIMLoss(cosim=False),
    MSSIMLoss(cosim=True),
    HSLuvLoss(criterion="charbonnier"),
    ColorLoss(criterion="charbonnier"),
    LumaLoss(criterion="charbonnier"),
    PerceptualLoss(
        layer_weights={
            "conv1_2": 0.1,
            "conv2_2": 0.1,
            "conv3_4": 1,
            "conv4_4": 1,
            "conv5_4": 1,
        },
        criterion="charbonnier",
    ),
    ADISTSLoss(),
    DISTSLoss(),
]

EPSILON = 1e-5  # torch.finfo(torch.float32).eps


class TestLossData(TypedDict):
    device: str
    black_image: Tensor
    dtype: torch.dtype


@pytest.fixture
def data() -> TestLossData:
    device = "cpu"
    input_shape = (1, 3, 16, 16)
    use_fp16 = False
    dtype = torch.float16 if use_fp16 else torch.float32
    return {
        "device": device,
        "dtype": dtype,
        "black_image": torch.zeros(input_shape, device=device, dtype=dtype),
    }


class TestLosses:
    @pytest.mark.parametrize("loss_class", [L1Loss, MSELoss, CharbonnierLoss])
    def test_pixellosses(self, loss_class: Callable) -> None:
        """Test loss: pixel losses"""

        pred = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        target = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        loss = loss_class(loss_weight=1.0, reduction="mean")
        out = loss(pred, target, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # -------------------- test with other reduction -------------------- #
        # reduction = none
        loss = loss_class(loss_weight=1.0, reduction="none")
        out = loss(pred, target, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 3, 4, 4)
        # test with spatial weights
        weight = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        out = loss(pred, target, weight=weight)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 3, 4, 4)

        # reduction = sum
        loss = loss_class(loss_weight=1.0, reduction="sum")
        out = loss(pred, target, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # -------------------- test unsupported loss reduction -------------------- #
        with pytest.raises(ValueError):
            loss_class(loss_weight=1.0, reduction="unknown")

    @pytest.mark.parametrize(
        "loss_fn",
        [pytest.param(loss_fn, id=f"{loss_fn}") for loss_fn in LOSS_FUNCTIONS],
    )
    def test_black_vs_black(self, data: TestLossData, loss_fn: nn.Module) -> None:
        loss_value = loss_fn(data["black_image"], data["black_image"])

        if type(loss_value) is tuple:
            assert loss_value[0] <= EPSILON
        else:
            assert loss_value <= EPSILON
