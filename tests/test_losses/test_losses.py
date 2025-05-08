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
from traiNNer.losses.cosim_loss import CosimLoss
from traiNNer.losses.dists_loss import DISTSLoss
from traiNNer.losses.ldl_loss import LDLLoss
from traiNNer.losses.mssim_loss import MSSIMLoss
from traiNNer.losses.perceptual_fp16_loss import (
    VGG19_CONV_LAYER_WEIGHTS,
    VGG19_RELU_LAYER_WEIGHTS,
)
from traiNNer.losses.perceptual_loss import PerceptualLoss

LOSS_FUNCTIONS = [
    L1Loss(
        1.0,
    ),
    CharbonnierLoss(
        1.0,
    ),
    MSELoss(
        1.0,
    ),
    MSSIMLoss(1.0),
    HSLuvLoss(1.0, criterion="charbonnier"),
    ColorLoss(1.0, criterion="charbonnier"),
    LumaLoss(1.0, criterion="charbonnier"),
    PerceptualLoss(
        1.0,
    ),
    ADISTSLoss(
        1.0,
    ),
    DISTSLoss(
        1.0,
    ),
    LDLLoss(
        1.0,
    ),
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
        if isinstance(loss_fn, LDLLoss):
            loss_value = loss_fn(
                data["black_image"], data["black_image"], data["black_image"]
            )
        else:
            loss_value = loss_fn(data["black_image"], data["black_image"])

        if type(loss_value) is tuple:
            assert loss_value[0] <= EPSILON
        elif isinstance(loss_value, dict):
            for k, v in loss_value.items():
                assert v <= EPSILON, k
        else:
            assert loss_value <= EPSILON

    @pytest.mark.parametrize(
        "criterion",
        ["pd+l1", "fd+l1", "pd", "fd", "l1", "charbonnier"],
    )
    def test_perceptual_loss(self, criterion: str) -> None:
        # Set random seed for reproducibility
        torch.manual_seed(42)

        batch_size = 1
        num_channels = 3
        height = 128
        width = 128

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generate random input tensors
        x = torch.randn((batch_size, num_channels, height, width), device=device)
        y = torch.randn((batch_size, num_channels, height, width), device=device)

        # Initialize both versions of the loss
        perceptual_loss_fn = PerceptualLoss(1.0, criterion=criterion).to(device=device)  # type: ignore

        if "pd" in criterion or "fd" in criterion:
            # relu layer weights
            assert (
                len(
                    set(VGG19_RELU_LAYER_WEIGHTS.keys())
                    - set(perceptual_loss_fn.vgg.stages.keys())
                )
                == 0
            )
        if "l1" in criterion or "charbonnier" in criterion:
            # conv layer weights
            assert (
                len(
                    set(VGG19_CONV_LAYER_WEIGHTS.keys())
                    - set(perceptual_loss_fn.vgg.stages.keys())
                )
                == 0
            )

        if "+" in criterion:
            assert len(perceptual_loss_fn.vgg.stages) == len(
                VGG19_CONV_LAYER_WEIGHTS
            ) + len(VGG19_RELU_LAYER_WEIGHTS)
        else:
            assert len(perceptual_loss_fn.vgg.stages) == len(VGG19_CONV_LAYER_WEIGHTS)

        # Compute losses
        _loss = perceptual_loss_fn(x, y)

    @pytest.mark.parametrize(
        "loss_fn",
        [pytest.param(loss_fn, id=f"{loss_fn}") for loss_fn in LOSS_FUNCTIONS],
    )
    def test_batch_size(self, data: TestLossData, loss_fn: nn.Module) -> None:
        pred = torch.rand(1, 3, 16, 16)
        gt = torch.rand(1, 3, 16, 16)

        if isinstance(loss_fn, LDLLoss):
            loss_value = loss_fn(pred, gt, gt)
        else:
            loss_value = loss_fn(pred, gt)

        pred2 = torch.cat([pred, pred], dim=0)
        gt2 = torch.cat([gt, gt], dim=0)

        if isinstance(loss_fn, LDLLoss):
            loss_value2 = loss_fn(pred2, gt2, gt2)
        else:
            loss_value2 = loss_fn(pred2, gt2)

        if isinstance(loss_value, dict):
            for key, val in loss_value.items():
                assert torch.allclose(val, loss_value2[key], atol=1e-6)
        else:
            assert torch.allclose(loss_value, loss_value2, atol=1e-6)

    def test_ssim(self) -> None:
        white = torch.ones(1, 3, 256, 256)
        black = torch.zeros(1, 3, 256, 256)
        gray128 = torch.full_like(black, 128 / 255)
        gray130 = torch.full_like(black, 130 / 255)
        white253 = torch.full_like(black, 253 / 255)
        black2 = torch.full_like(black, 2 / 255)
        white222 = torch.full_like(black, 222 / 255)
        black26 = torch.full_like(black, 26 / 255)

        loss_fn = MSSIMLoss(1.0, include_luminance=False)
        hsluv_fn = HSLuvLoss(1.0)

        print(loss_fn(white253, white))  # should be near 0
        print(loss_fn(gray128, gray130))  # should be near 0
        print(loss_fn(black, black2))  # should be near 0 but mssim fails
        print(loss_fn(white222, white))  # shouldn't be near 0 but mssim fails
        print(loss_fn(black, black26))  # should be near 0 but mssim fails

        print(hsluv_fn(white253, white))
        print(hsluv_fn(gray128, gray130))
        print(hsluv_fn(black, black2))
        print(hsluv_fn(white222, white))
        print(hsluv_fn(black, black26))

    def test_cosim(self) -> None:
        pred = torch.rand(1, 3, 16, 16)
        gt = torch.rand(1, 3, 16, 16)

        mssim_fn = MSSIMLoss(1.0)
        cosim_fn = CosimLoss(1.0, cosim_lambda=10)

        print(mssim_fn(pred, gt))
        print(cosim_fn(pred, gt))

    def test_msssimlum(self) -> None:
        white = torch.ones(1, 3, 256, 256)
        black = torch.zeros(1, 3, 256, 256)
        gray128 = torch.full_like(black, 128 / 255)
        gray130 = torch.full_like(black, 130 / 255)
        white253 = torch.full_like(black, 253 / 255)
        black2 = torch.full_like(black, 2 / 255)
        white222 = torch.full_like(black, 222 / 255)
        black26 = torch.full_like(black, 26 / 255)
        pred = black26
        gt = black
        # pred = torch.rand(1, 3, 256, 256)
        # gt = torch.rand(1, 3, 256, 256)

        mssimlum_fn = MSSIMLoss(1.0, include_luminance=True)
        mssimnolum_fn = MSSIMLoss(1.0, include_luminance=False)

        print(mssimlum_fn(pred, gt))
        print(mssimnolum_fn(pred, gt))
