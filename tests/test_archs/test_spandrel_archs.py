import os
import sys
from collections.abc import Callable
from typing import TypedDict

import pytest
import torch
from torch import Tensor, nn

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r"..\.."))
)
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.losses.basic_loss import L1Loss

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


class TestArchData(TypedDict):
    device: str
    lq: Tensor
    dtype: torch.dtype


@pytest.fixture
def data() -> TestArchData:
    device = "cpu"
    input_shape = (2, 3, 32, 32)
    use_fp16 = False
    dtype = torch.float16 if use_fp16 else torch.float32
    return {
        "device": device,
        "dtype": dtype,
        "lq": torch.rand(input_shape, device=device, dtype=dtype),
    }


class TestArchs:
    @pytest.mark.parametrize(
        "arch",
        [pytest.param(arch, id=f"test_{name}") for name, arch in FILTERED_REGISTRY],
    )
    def test_arch_inference(
        self, data: TestArchData, arch: Callable[..., nn.Module]
    ) -> None:
        device = data["device"]
        lq = data["lq"]
        dtype = data["dtype"]
        scale = 5
        model = arch(scale=scale).eval().to(device, dtype=dtype)

        with torch.inference_mode():
            output = model(lq)
            assert (
                output.shape[2] == lq.shape[2] * scale
                and output.shape[3] == lq.shape[3] * scale
            )

    @pytest.mark.parametrize(
        "arch",
        [pytest.param(arch, id=f"train_{name}") for name, arch in FILTERED_REGISTRY],
    )
    def test_arch_training(
        self, data: TestArchData, arch: Callable[..., nn.Module]
    ) -> None:
        device = data["device"]
        scale = 4
        lq = data["lq"]
        gt_shape = (lq.shape[0], lq.shape[1], lq.shape[2] * scale, lq.shape[3] * scale)
        dtype = data["dtype"]
        random_gt = torch.rand(gt_shape, device=device, dtype=dtype)
        model = arch(scale=scale).train().to(device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = L1Loss()

        output = model(lq)
        l_g_l1 = loss_fn(output, random_gt)
        l_g_l1.backward()
        optimizer.step()
