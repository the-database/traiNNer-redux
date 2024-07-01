import itertools
from collections.abc import Callable
from typing import TypedDict

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.losses.basic_loss import L1Loss

# A list of archs which should be excluded from testing. dat, hat, and swinir
# testing is covered via their preset variants such as dat_2 or swinir_m.
# VGG and UNet are not applicable to the tests in this test suite and may
# require their own test cases.
EXCLUDE_ARCHS = {
    "dat",
    "hat",
    "swinir",
    "vggstylediscriminator",
    "unetdiscriminatorsn_traiNNer",
    "vggfeatureextractor",
}
FILTERED_REGISTRY = [
    (name, arch)
    for name, arch in list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
    if name not in EXCLUDE_ARCHS
]

ALL_SCALES = [1, 2, 3, 4]

# A list of tuples in the format of (name, arch, scale).
FILTERED_REGISTRIES_SCALES = [
    (*a, b) for a, b in itertools.product(FILTERED_REGISTRY, ALL_SCALES)
]

# A dict of archs mapped to a list of scales that arch doesn't support.
EXCLUDE_ARCH_SCALES = {"swinir_l": [3], "realcugan": [1]}

# A set of arch names whose arch requires a minimum batch size of 2 in order to train.
REQUIRE_BATCH_2 = {"dat_2"}

# A set of arch names whose arch requires a minimum
# image size of 32x32 to do training or inference with.
REQUIRE_32_HW = {"realcugan"}


class TestArchData(TypedDict):
    device: str
    lq: Tensor
    dtype: torch.dtype


@pytest.fixture
def data() -> TestArchData:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shape = (1, 3, 16, 16)
    use_fp16 = False
    dtype = torch.float16 if use_fp16 else torch.float32
    return {
        "device": device,
        "dtype": dtype,
        "lq": torch.rand(input_shape, device=device, dtype=dtype),
    }


class TestArchs:
    @pytest.mark.parametrize(
        "name,arch,scale",
        [
            pytest.param(name, arch, scale, id=f"test_{name}_{scale}x")
            for name, arch, scale in FILTERED_REGISTRIES_SCALES
        ],
    )
    def test_arch_inference(
        self,
        data: TestArchData,
        name: str,
        arch: Callable[..., nn.Module],
        scale: int,
    ) -> None:
        if name in EXCLUDE_ARCH_SCALES and scale in EXCLUDE_ARCH_SCALES[name]:
            pytest.skip(f"Skipping known unsupported {scale}x scale for {name}")

        device = data["device"]
        lq = data["lq"]
        dtype = data["dtype"]
        model = arch(scale=scale).eval().to(device, dtype=dtype)

        # For performance reasons, we try to use 1x3x16x16 input tensors by default,
        # but this is too small for some networks. Double the resolution to 1x3x32x32
        # for any networks that require a larger input size.
        if name in REQUIRE_32_HW:
            lq = F.interpolate(lq, scale_factor=2)

        with torch.inference_mode():
            output = model(lq)
            assert (
                output.shape[2] == lq.shape[2] * scale
                and output.shape[3] == lq.shape[3] * scale
            ), f"{name}: {output.shape} is not {scale}x {lq.shape}"

    @pytest.mark.parametrize(
        "name,arch,scale",
        [
            pytest.param(name, arch, scale, id=f"train_{name}_{scale}x")
            for name, arch, scale in FILTERED_REGISTRIES_SCALES
        ],
    )
    def test_arch_training(
        self,
        data: TestArchData,
        name: str,
        arch: Callable[..., nn.Module],
        scale: int,
    ) -> None:
        if name in EXCLUDE_ARCH_SCALES and scale in EXCLUDE_ARCH_SCALES[name]:
            pytest.skip(f"Skipping known unsupported {scale}x scale for {name}")

        device = data["device"]
        lq = data["lq"]

        # Some networks can't be trained with a batch size of 1.
        # Duplicate the input to set batch 1 to batch 2 for these networks.
        if name in REQUIRE_BATCH_2:
            lq = lq.repeat(2, 1, 1, 1)

        # For performance reasons, we try to use 1x3x16x16 input tensors by default,
        # but this is too small for some networks. Double the resolution to 1x3x32x32
        # for any networks that require a larger input size.
        if name in REQUIRE_32_HW:
            lq = F.interpolate(lq, scale_factor=2)

        gt_shape = (lq.shape[0], lq.shape[1], lq.shape[2] * scale, lq.shape[3] * scale)
        dtype = data["dtype"]
        gt = torch.rand(gt_shape, device=device, dtype=dtype)
        model = arch(scale=scale).train().to(device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = L1Loss()

        output = model(lq)
        assert not torch.isnan(output).any(), "NaN detected in model output"

        l_g_l1 = loss_fn(output, gt)
        assert not torch.isnan(l_g_l1).any(), "NaN detected in loss"

        l_g_l1.backward()

        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(
                    param.grad
                ).any(), f"NaN detected in gradients of parameter {param}"

        optimizer.step()

        for param in model.parameters():
            assert not torch.isnan(
                param
            ).any(), f"NaN detected in parameter {param} after optimizer step"
