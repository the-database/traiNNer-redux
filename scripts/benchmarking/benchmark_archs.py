import os
import sys
import time
from typing import Any

import torch
from torch import Tensor, nn

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r"..\.."))
)
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16, OFFICIAL_METRICS

# ALL_REGISTRIES = list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
ALL_REGISTRIES = list(ARCH_REGISTRY)
EXCLUDE_BENCHMARK_ARCHS = {
    "dct",
    "dunet",
    "eimn",
    "hat",
    "swinir",
    "swin2sr",
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
ALL_SCALES = [4, 3, 2, 1]
LIGHTWEIGHT_ARCHS = {
    "cfsr",
    "realcugan",
    "span",
    "compact",
    "plksr_tiny",
    "ultracompact",
    "superultracompact",
    "spanplus",
    "spanplus_s",
    "spanplus_st",
    "spanplus_sts",
}
# For archs that have extra parameters, list all combinations that need to be benchmarked.
EXTRA_ARCH_PARAMS: dict[str, list[dict[str, Any]]] = {
    k: [] for k, _ in FILTERED_REGISTRY
}
EXTRA_ARCH_PARAMS["realplksr"] = [
    {"upsampler": "dysample", "layer_norm": False},
    {"upsampler": "pixelshuffle", "layer_norm": False},
    {"upsampler": "dysample", "layer_norm": True},
    {"upsampler": "pixelshuffle", "layer_norm": True},
]

EXTRA_ARCH_PARAMS["realplksrmod"] = [
    {"upsampler": "dysample"},
    {"upsampler": "pixelshuffle"},
]

EXTRA_ARCH_PARAMS["esrgan"] = [
    {"use_pixel_unshuffle": True},
    {"use_pixel_unshuffle": False},
]

# A list of tuples in the format of (name, arch, extra_params).
FILTERED_REGISTRIES_PARAMS = [
    (name, arch, extra_params)
    for (name, arch) in FILTERED_REGISTRY
    for extra_params in (EXTRA_ARCH_PARAMS[name] if EXTRA_ARCH_PARAMS[name] else [{}])
]

# A dict of archs mapped to a list of scale + arch params that arch doesn't support.
EXCLUDE_ARCH_SCALES = {
    "swinir_l": [{"scale": 3, "extra_arch_params": {}}],
    "realcugan": [{"scale": 1, "extra_arch_params": {}}],
}


def format_extra_params(extra_arch_params: dict[str, Any]) -> str:
    out = ""

    for k, v in extra_arch_params.items():
        if isinstance(v, str):
            out += f"{v} "
        else:
            out += f"{k}={v} "

    return out.strip()


def get_line(
    name: str,
    dtype_str: str,
    avg_time: float,
    fps: float,
    vram: float,
    params: int,
    scale: int,
    extra_arch_params: dict[str, Any],
    print_markdown: bool = False,
) -> str:
    name_separator = "|" if print_markdown else ": "
    separator = "|" if print_markdown else ",    "
    edge_separator = "|" if print_markdown else ""
    unsupported_value = "-"
    name_str = f"{name} {format_extra_params(extra_arch_params)} {scale}x {dtype_str}"

    fps_label = "" if print_markdown else "FPS: "
    sec_img_label = "" if print_markdown else "sec/img: "
    vram_label = "" if print_markdown else "VRAM: "
    params_label = "" if print_markdown else "Params: "
    psnrdf2k_label = "" if print_markdown else "PSNR (DF2K): "
    ssimdf2k_label = "" if print_markdown else "SSIM (DF2K): "
    psnrdiv2k_label = "" if print_markdown else "PSNR (DIV2K): "
    ssimdiv2k_label = "" if print_markdown else "SSIM (DIV2K): "

    psnrdf2k = format(unsupported_value, "<5s")
    ssimdf2k = format(unsupported_value, "<6s")
    psnrdiv2k = format(unsupported_value, "<5s")
    ssimdiv2k = format(unsupported_value, "<6s")

    if name in OFFICIAL_METRICS:
        if scale in OFFICIAL_METRICS[name]:
            if "df2k_psnr" in OFFICIAL_METRICS[name][scale]:
                psnrdf2k = format(OFFICIAL_METRICS[name][scale]["df2k_psnr"], ".2f")
            if "df2k_ssim" in OFFICIAL_METRICS[name][scale]:
                ssimdf2k = format(OFFICIAL_METRICS[name][scale]["df2k_ssim"], ".4f")
            if "div2k_psnr" in OFFICIAL_METRICS[name][scale]:
                psnrdiv2k = format(OFFICIAL_METRICS[name][scale]["div2k_psnr"], ".2f")
            if "div2k_ssim" in OFFICIAL_METRICS[name][scale]:
                ssimdiv2k = format(OFFICIAL_METRICS[name][scale]["div2k_ssim"], ".4f")

    if params != -1:
        return f"{edge_separator}{name_str:<35}{name_separator}{fps_label}{fps:>8.2f}{separator}{sec_img_label}{avg_time:>8.4f}{separator}{vram_label}{vram:>8.2f} GB{separator}{params_label}{params:>10,d}{separator}{psnrdf2k_label}{psnrdf2k}{separator}{ssimdf2k_label}{ssimdf2k}{separator}{psnrdiv2k_label}{psnrdiv2k}{separator}{ssimdiv2k_label}{ssimdiv2k}{edge_separator}"

    return f"{edge_separator}{name_str:<35}{name_separator}{fps_label}{unsupported_value:<8}{separator}{sec_img_label}{unsupported_value:<8}{separator}{vram_label}{unsupported_value:<8}{separator}{params_label}{unsupported_value:<10}{separator}{psnrdf2k_label}{unsupported_value}{separator}{ssimdf2k_label}{unsupported_value}{separator}{psnrdiv2k_label}{unsupported_value}{separator}{ssimdiv2k_label}{edge_separator}"


def benchmark_model(
    model: nn.Module, input_tensor: Tensor, warmup_runs: int = 5, num_runs: int = 10
) -> tuple[float, Tensor]:
    for _ in range(warmup_runs):
        with torch.inference_mode():
            model(input_tensor)

    output = None
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.inference_mode():
            output = model(input_tensor)
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    assert output is not None
    return avg_time, output


def get_dtype(name: str, use_amp: bool) -> tuple[str, torch.dtype]:
    amp_bf16 = name in ARCHS_WITHOUT_FP16
    dtype_str = "fp32" if not use_amp else ("bf16" if amp_bf16 else "fp16")
    dtype = (
        torch.float32 if not use_amp else torch.bfloat16 if amp_bf16 else torch.float16
    )
    return dtype_str, dtype


if __name__ == "__main__":
    start_script_time = time.time()
    device = "cuda"

    input_shape = (1, 3, 480, 640)

    warmup_runs = 1  # 1
    num_runs = 5  # 5
    lightweight_num_runs = 250
    print_markdown = True
    n, c, h, w = input_shape
    results_by_scale: dict[
        int, list[tuple[str, str, float, float, float, int, int, dict[str, Any]]]
    ] = {}
    results_by_arch: dict[
        str, dict[int, tuple[str, str, float, float, float, int, int, dict[str, Any]]]
    ] = {}

    for use_amp in [True]:
        for scale in ALL_SCALES:
            results_by_scale[scale] = []

            for name, arch, extra_arch_params in FILTERED_REGISTRIES_PARAMS:
                arch_key = f"{name} {format_extra_params(extra_arch_params)}"
                dtype_str, dtype = get_dtype(name, use_amp)
                try:
                    if "rcan" not in name:
                        continue
                    random_input = torch.rand(
                        input_shape,
                        device=device,
                        # dtype=dtype,
                    ).to(
                        memory_format=torch.channels_last,
                        non_blocking=True,
                    )
                    model = (
                        arch(scale=scale, **extra_arch_params)
                        .eval()
                        .to(
                            device,
                            # dtype=dtype,
                            memory_format=torch.channels_last,
                            non_blocking=True,
                        )
                    )

                    if arch_key not in results_by_arch:
                        results_by_arch[arch_key] = {}
                    total_params = sum(p[1].numel() for p in model.named_parameters())
                    runs = (
                        lightweight_num_runs if name in LIGHTWEIGHT_ARCHS else num_runs
                    )
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    with torch.autocast(
                        device_type="cuda",
                        dtype=dtype,
                        enabled=use_amp,
                    ):
                        avg_time, output = benchmark_model(
                            model, random_input, warmup_runs, runs
                        )

                    if not (
                        output.shape[2] == random_input.shape[2] * scale
                        and output.shape[3] == random_input.shape[3] * scale
                    ):
                        msg = f"{name}: {output.shape} is not {scale}x {random_input.shape}"
                        print(msg)
                        raise ValueError(msg)

                    vram_usage = torch.cuda.max_memory_allocated(device) / (1024**3)
                    row = (
                        name,
                        dtype_str,
                        avg_time,
                        1 / avg_time,
                        vram_usage,
                        total_params,
                        scale,
                        extra_arch_params,
                    )

                    results_by_scale[scale].append(row)
                    results_by_arch[arch_key][scale] = row
                except Exception as e:
                    import traceback

                    traceback.print_exception(e)
                    row = (
                        name,
                        dtype_str,
                        float("inf"),
                        float("inf"),
                        float("inf"),
                        -1,
                        scale,
                        extra_arch_params,
                    )
                    results_by_scale[scale].append(row)
                    results_by_arch[arch_key][scale] = row
                print(get_line(*results_by_scale[scale][-1]))

            results_by_scale[scale].sort(key=lambda x: x[2])

    print("## By Scale")

    for scale in ALL_SCALES:
        if print_markdown:
            print(f"\n### {scale}x scale")
            print(
                f"{input_shape} input, {warmup_runs} warmup + {num_runs} runs averaged"
            )
            print(
                "|Name|FPS|sec/img|VRAM|Params|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)"
            )
            print("|:-|-:|-:|-:|-:|-:|-:|-:|-:|")
        for row in results_by_scale[scale]:
            print(get_line(*row, print_markdown))

    if print_markdown:
        print("\n## By Architecture")

    for arch_name in sorted(results_by_arch.keys()):
        print(f"\n### {arch_name}")
        runs = lightweight_num_runs if arch_name in LIGHTWEIGHT_ARCHS else num_runs
        print(f"{input_shape} input, {warmup_runs} warmup + {runs} runs averaged")
        print(
            "|Name|FPS|sec/img|VRAM|Params|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)"
        )
        print("|:-|-:|-:|-:|-:|-:|-:|-:|-:|")
        for scale in ALL_SCALES:
            print(get_line(*results_by_arch[arch_name][scale], print_markdown))

    end_script_time = time.time()
    print(f"\nFinished in: {end_script_time - start_script_time:.2f} seconds")
