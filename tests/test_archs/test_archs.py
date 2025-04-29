from collections.abc import Callable
from typing import Any, TypedDict

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
    "autoencoder",
    "dunet",
    "eimn",
    "hat",
    "lmlt",
    "metagan2",
    "swin2sr",
    "swinir",
    "unetdiscriminatorsn",
    "vggfeatureextractor",
    "vggstylediscriminator",
}
FILTERED_REGISTRY = [
    (name, arch)
    for name, arch in list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
    if name not in EXCLUDE_ARCHS
]

ALL_SCALES = [1, 2, 3, 4]

# For archs that have extra parameters, list all combinations that need to be tested.
EXTRA_ARCH_PARAMS: dict[str, list[dict[str, Any]]] = {
    k: [] for k, _ in FILTERED_REGISTRY
}
EXTRA_ARCH_PARAMS["realplksr"] = [
    {"upsampler": "dysample"},
    {"upsampler": "pixelshuffle"},
]

# TODO remove after finalizing
EXTRA_ARCH_PARAMS["realplksrmod"] = EXTRA_ARCH_PARAMS["realplksr"]

EXTRA_ARCH_PARAMS["mosr"] = [
    {"upsampler": "dysample"},
    {"upsampler": "pixelshuffle"},
    {"upsampler": "geoensemblepixelshuffle"},
]

EXTRA_ARCH_PARAMS["mosr_t"] = [
    {"upsampler": "dysample"},
    {"upsampler": "pixelshuffle"},
    {"upsampler": "geoensemblepixelshuffle"},
]

EXTRA_ARCH_PARAMS["rcan"] = [
    {"unshuffle_mod": True},
    {"unshuffle_mod": False},
]

# A list of tuples in the format of (name, arch, scale, extra_params).
FILTERED_REGISTRIES_SCALES_PARAMS = [
    (name, arch, scale, extra_params)
    for (name, arch) in FILTERED_REGISTRY
    for scale in ALL_SCALES
    for extra_params in (EXTRA_ARCH_PARAMS[name] if EXTRA_ARCH_PARAMS[name] else [{}])
]

# A dict of archs mapped to a list of scale + arch params that arch doesn't support.
EXCLUDE_ARCH_SCALES = {
    "swinir_l": [{"scale": 3, "extra_arch_params": {}}],
    "swin2sr_l": [
        {"scale": 1, "extra_arch_params": {}},
        {"scale": 2, "extra_arch_params": {}},
        {"scale": 3, "extra_arch_params": {}},
    ],
    "realcugan": [{"scale": 1, "extra_arch_params": {}}],
    "tscunet": [{"scale": 3, "extra_arch_params": {}}],
    "scunet_aaf6aa": [{"scale": 3, "extra_arch_params": {}}],
    "rtmosr_l": [{"scale": 3, "extra_arch_params": {}}],
    "rtmosr_ul": [{"scale": 3, "extra_arch_params": {}}],
    "cascadedgaze": [
        {"scale": 2, "extra_arch_params": {}},
        {"scale": 3, "extra_arch_params": {}},
        {"scale": 4, "extra_arch_params": {}},
    ],
    "unetsegmentation": [
        {"scale": 2, "extra_arch_params": {}},
        {"scale": 3, "extra_arch_params": {}},
        {"scale": 4, "extra_arch_params": {}},
    ],
}

# A set of arch names whose arch requires a minimum batch size of 2 in order to train.
REQUIRE_BATCH_2 = {"dat_2", "dat", "dat_light", "dat_s"}
ADD_VSR_DIM = {"tscunet"}

# A set of arch names whose arch requires a minimum
# image size of 32x32 to do training or inference with.
REQUIRE_32_HW = {"dwt", "emt", "hit_srf", "realcugan"}
REQUIRE_64_HW = {
    "cascadedgaze",
    "hit_lmlt",
    "hit_sir",
    "hit_sng",
    "lmlt_base",
    "lmlt_large",
    "lmlt_tiny",
    "metaflexnet",
    "scunet_aaf6aa",
    "tscunet",
    "swin2sr_l",
    "swin2sr_m",
    "swin2sr_s",
}


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
        "name,arch,scale,extra_arch_params",
        [
            pytest.param(
                name,
                arch,
                scale,
                extra_arch_params,
                id=f"test_{name}_{scale}x_{extra_arch_params}",
            )
            for name, arch, scale, extra_arch_params in FILTERED_REGISTRIES_SCALES_PARAMS
        ],
    )
    def test_arch_inference(
        self,
        data: TestArchData,
        name: str,
        arch: Callable[..., nn.Module],
        scale: int,
        extra_arch_params: dict[str, Any],
    ) -> None:
        if (
            name in EXCLUDE_ARCH_SCALES
            and {"scale": scale, "extra_arch_params": extra_arch_params}
            in EXCLUDE_ARCH_SCALES[name]
        ):
            pytest.skip(f"Skipping known unsupported {scale}x scale for {name}")

        device = data["device"]
        lq = data["lq"]
        dtype = data["dtype"]
        model = arch(scale=scale, **extra_arch_params).eval().to(device, dtype=dtype)

        # For performance reasons, we try to use 1x3x16x16 input tensors by default,
        # but this is too small for some networks. Double the resolution to 1x3x32x32
        # for any networks that require a larger input size.
        if name in REQUIRE_64_HW:
            lq = F.interpolate(lq, scale_factor=4)
        elif name in REQUIRE_32_HW:
            lq = F.interpolate(lq, scale_factor=2)

        if name in ADD_VSR_DIM:
            lq = lq.repeat(5, 1, 1, 1).unsqueeze(0)
            # assert False, lq.shape

        with torch.inference_mode():
            output = model(lq)
            assert (
                output.shape[-2] == lq.shape[-2] * scale
                and output.shape[-1] == lq.shape[-1] * scale
            ), f"{name}: {output.shape} is not {scale}x {lq.shape}"

    @pytest.mark.parametrize(
        "name,arch,scale,extra_arch_params",
        [
            pytest.param(
                name,
                arch,
                scale,
                extra_arch_params,
                id=f"train_{name}_{scale}x_{extra_arch_params}",
            )
            for name, arch, scale, extra_arch_params in FILTERED_REGISTRIES_SCALES_PARAMS
        ],
    )
    def test_arch_training(
        self,
        data: TestArchData,
        name: str,
        arch: Callable[..., nn.Module],
        scale: int,
        extra_arch_params: dict[str, Any],
    ) -> None:
        if (
            name in EXCLUDE_ARCH_SCALES
            and {"scale": scale, "extra_arch_params": extra_arch_params}
            in EXCLUDE_ARCH_SCALES[name]
        ):
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
        if name in REQUIRE_64_HW:
            lq = F.interpolate(lq, scale_factor=4)
        elif name in REQUIRE_32_HW:
            lq = F.interpolate(lq, scale_factor=2)

        if name in ADD_VSR_DIM:
            lq = lq.repeat(5, 1, 1, 1).unsqueeze(0)
            # assert False, lq.shape

        gt_shape = (
            lq.shape[0],
            lq.shape[-3],
            lq.shape[-2] * scale,
            lq.shape[-1] * scale,
        )
        dtype = data["dtype"]
        gt = torch.rand(gt_shape, device=device, dtype=dtype)
        model = arch(scale=scale, **extra_arch_params).train().to(device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = L1Loss(1.0)

        output = model(lq)
        assert output.shape == gt.shape
        assert not torch.isnan(output).any(), "NaN detected in model output"

        l_g_l1 = loss_fn(output, gt)
        assert not torch.isnan(l_g_l1).any(), "NaN detected in loss"

        l_g_l1.backward()

        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN detected in gradients of parameter {param}"
                )

        optimizer.step()

        for param in model.parameters():
            assert not torch.isnan(param).any(), (
                f"NaN detected in parameter {param} after optimizer step"
            )

    # @pytest.mark.parametrize(
    #     "name,arch,scale,extra_arch_params",
    #     [
    #         pytest.param(
    #             name,
    #             arch,
    #             scale,
    #             extra_arch_params,
    #             id=f"test_{name}_{scale}x_{extra_arch_params}",
    #         )
    #         for name, arch, scale, extra_arch_params in FILTERED_REGISTRIES_SCALES_PARAMS
    #     ],
    # )
    # def test_arch_channels_last(
    #     self,
    #     data: TestArchData,
    #     name: str,
    #     arch: Callable[..., nn.Module],
    #     scale: int,
    #     extra_arch_params: dict[str, Any],
    # ):
    #     lq = data["lq"].to(device="cuda", memory_format=torch.channels_last)
    #     dtype = data["dtype"]
    #     device = data["device"]
    #     model: nn.Module = (
    #         arch(scale=scale, **extra_arch_params)
    #         .eval()
    #         .to(device=device, dtype=dtype, memory_format=torch.channels_last)
    #     )  # type: ignore

    #     assert lq.is_contiguous(memory_format=torch.channels_last)
    #     print(list(model.parameters())[0].stride())

    #     with torch.inference_mode(), torch.autocast(dtype=dtype, device_type=device):
    #         out = model(lq)
    #         assert out.is_contiguous(memory_format=torch.channels_last)
