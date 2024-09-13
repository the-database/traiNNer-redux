from typing import Any

import torch
from torch import autocast, nn
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.losses.basic_loss import L1Loss

ALL_REGISTRIES = list(ARCH_REGISTRY)
EXCLUDE_BENCHMARK_ARCHS = {
    "dat",
    "hat",
    "swinir",
    "lmlt",
    "vggstylediscriminator",
    "unetdiscriminatorsn_traiNNer",
    "vggfeatureextractor",
}

FILTERED_REGISTRY = [
    (name, arch)
    for name, arch in list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
    if name not in EXCLUDE_BENCHMARK_ARCHS
]
# For archs that have extra parameters, list all combinations that need to be benchmarked.
EXTRA_ARCH_PARAMS: dict[str, list[dict[str, Any]]] = {
    k: [] for k, _ in FILTERED_REGISTRY
}
EXTRA_ARCH_PARAMS["realplksr"] = [
    {"upsampler": "dysample"},
    {"upsampler": "pixelshuffle"},
]
# A list of tuples in the format of (name, arch, extra_params).
FILTERED_REGISTRIES_PARAMS = [
    (name, arch, extra_params)
    for (name, arch) in FILTERED_REGISTRY
    for extra_params in (EXTRA_ARCH_PARAMS[name] if EXTRA_ARCH_PARAMS[name] else [{}])
]


# Function to check if model supports fp16
def supports_fp16(model, input_tensor):
    try:
        with autocast(dtype=torch.float16, device_type="cuda"):
            output = model(input_tensor.half())
        return True
    except Exception as e:
        print(f"FP16 support failed: {e}")
        return False


# Function to test inference performance in bf16 vs fp16 compared to fp32
def compare_precision(model, input_tensor, criterion):
    # Compute fp32 output as the baseline
    with torch.no_grad():
        fp32_output = model(input_tensor)

    # Test fp16 inference
    fp16_loss = None
    try:
        with autocast(dtype=torch.float16, device_type="cuda"):
            fp16_output = model(input_tensor)
            print(fp16_output)
        fp16_loss = criterion(fp16_output.float(), fp32_output).item()
    except Exception as e:
        print(f"Error in FP16 inference: {e}")
        fp16_loss = float("inf")

    # Test bf16 inference
    bf16_loss = None
    try:
        with autocast(dtype=torch.bfloat16, device_type="cuda"):
            bf16_output = model(input_tensor)
        bf16_loss = criterion(bf16_output.float(), fp32_output).item()
    except Exception as e:
        print(f"Error in BF16 inference: {e}")
        bf16_loss = float("inf")

    return fp16_loss, bf16_loss


if __name__ == "__main__":
    for name, arch, extra_arch_params in FILTERED_REGISTRIES_PARAMS:
        # Example model and input setup
        # model = SPAN(num_in_ch=3, num_out_ch=3)
        try:
            if name != "dat_2":
                continue

            net: nn.Module = arch(scale=4, **extra_arch_params).eval().to("cuda")
            net.load_state_dict(
                torch.load(
                    r"C:\Users\jsoos\Documents\programming\DAT\experiments\pretrained_models\DAT-2\DAT_2_x4.pth",
                    weights_only=True,
                )["params"]
            )

            input_tensor = torch.randn((1, 3, 64, 64), device="cuda")
            criterion = L1Loss()

            # Compare inference performance in fp16 and bf16 to fp32
            fp16_loss, bf16_loss = compare_precision(net, input_tensor, criterion)

            if fp16_loss < bf16_loss:
                print(
                    f"{name}: FP16 is closer to FP32 with loss {fp16_loss} vs BF16 loss {bf16_loss}"
                )
            elif bf16_loss < fp16_loss:
                print(
                    f"{name}: BF16 is closer to FP32 with loss {bf16_loss} vs FP16 loss {fp16_loss}"
                )
            else:
                print(
                    f"{name}: Both have similar performance: FP16 loss {fp16_loss}, BF16 loss {bf16_loss}"
                )
        except Exception as e:
            print(f"skip {name}", e)
