import itertools
from collections.abc import Callable
from typing import TypedDict

import pytest
import torch
from torch import Tensor, nn
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

ALL_SCALES = [1, 2, 3, 4]

FILTERED_REGISTRIES_SCALES = [
    (*a, b) for a, b in itertools.product(FILTERED_REGISTRY, ALL_SCALES)
]

EXCLUDE_ARCH_SCALES = {"swinir_l": [3], "realcugan": [1]}


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
