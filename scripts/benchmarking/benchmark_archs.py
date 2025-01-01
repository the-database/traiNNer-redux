import os
import sys
import time
from collections.abc import Callable
from io import TextIOWrapper
from typing import Any

import torch
from torch import Tensor, memory_format, nn

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r"..\.."))
)
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16, OFFICIAL_METRICS

# ALL_REGISTRIES = list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
ALL_REGISTRIES = list(ARCH_REGISTRY)
EXCLUDE_BENCHMARK_ARCHS = {
    "artcnn",
    "dct",
    "dunet",
    "eimn",
    "hat",
    "swinir",
    "swin2sr",
    "lmlt",
    "vggstylediscriminator",
    "unetdiscriminatorsn",
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


def printfc(text: str, f: TextIOWrapper) -> None:
    print(text)
    f.write(f"{text}\n")


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
    fps_channels_last: float,
    channels_last_vs_baseline: float,
    print_markdown: bool = False,
) -> str:
    name_separator = "|" if print_markdown else ": "
    separator = "|" if print_markdown else ",    "
    edge_separator = "|" if print_markdown else ""
    unsupported_value = "-"
    name_str = f"{name} {format_extra_params(extra_arch_params)} {scale}x {dtype_str}"

    fps_label = "" if print_markdown else "FPS: "
    fps_cl_label = "" if print_markdown else "FPS (channels last): "
    channels_last_vs_label = "" if print_markdown else "Channels last vs baseline: "
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
        return f"{edge_separator}{name_str:<35}{name_separator}{fps_label}{fps:>8.2f}{separator}{fps_cl_label}{fps_channels_last:>8.2f}{separator}{channels_last_vs_label}{channels_last_vs_baseline:>1.2f}x{separator}{sec_img_label}{avg_time:>8.4f}{separator}{vram_label}{vram:>8.2f} GB{separator}{psnrdf2k_label}{psnrdf2k}{separator}{ssimdf2k_label}{ssimdf2k}{separator}{psnrdiv2k_label}{psnrdiv2k}{separator}{ssimdiv2k_label}{ssimdiv2k}{separator}{params_label}{params:>10,d}{edge_separator}"

    return f"{edge_separator}{name_str:<35}{name_separator}{fps_label}{unsupported_value:<8}{separator}{fps_cl_label}{unsupported_value:<8}{separator}{channels_last_vs_label}{unsupported_value:<8}{separator}{sec_img_label}{unsupported_value:<8}{separator}{vram_label}{unsupported_value:<8}{separator}{psnrdf2k_label}{unsupported_value}{separator}{ssimdf2k_label}{unsupported_value}{separator}{psnrdiv2k_label}{unsupported_value}{separator}{ssimdiv2k_label}{separator}{params_label}{unsupported_value:<10}{edge_separator}"


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


def benchmark_arch(
    name: str, arch: Callable, extra_arch_params: dict, memory_format: memory_format
) -> tuple:
    random_input = torch.rand(
        input_shape,
        device=device,
        # dtype=dtype,
    ).to(
        memory_format=memory_format,
        non_blocking=True,
    )
    model = (
        arch(scale=scale, **extra_arch_params)
        .eval()
        .to(
            device,
            # dtype=dtype,
            memory_format=memory_format,
            non_blocking=True,
        )
    )

    total_params = sum(p[1].numel() for p in model.named_parameters())
    runs = lightweight_num_runs if name in LIGHTWEIGHT_ARCHS else num_runs
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=use_amp,
    ):
        avg_time, output = benchmark_model(model, random_input, warmup_runs, runs)

    if not (
        output.shape[2] == random_input.shape[2] * scale
        and output.shape[3] == random_input.shape[3] * scale
    ):
        msg = f"{name}: {output.shape} is not {scale}x {random_input.shape}"
        print(msg)
        # raise ValueError(msg)  # TODO restore

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

    return row


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
        int,
        list[
            tuple[str, str, float, float, float, int, int, dict[str, Any], float, float]
        ],
    ] = {}
    results_by_arch: dict[
        str,
        dict[
            int,
            tuple[
                str, str, float, float, float, int, int, dict[str, Any], float, float
            ],
        ],
    ] = {}

    with open("docs/source/benchmarks.md", "w") as f:
        f.write("""# PyTorch Inference Benchmarks by Architecture (AMP & channels last)

All benchmarks were generated using [benchmark_archs.py](https://github.com/the-database/traiNNer-redux/blob/master/scripts/benchmarking/benchmark_archs.py). The benchmarks were done on a Windows 11 PC with RTX 4090 + i9-13000K.

Note that these benchmarks only measure the raw inference step of these architectures. In practice, several other factors may contribute to results not matching the benchmarks shown here. For example, when comparing two architectures with the same inference speed but one has double the VRAM usage, the one with less VRAM usage will be faster to upscale with for larger images, because the one with higher VRAM usage would require tiling to avoid running out of VRAM in order to upscale a large image while the one with lower VRAM usage could upscale the entire image at once without tiling.

PSNR and SSIM scores are a rough measure of quality, higher is better. These scores should not be taken as an absolute that one architecture is better than another. Metrics are calculated using the officially released models optimized on L1 loss, and are trained on either the DF2K or DIV2K training dataset. When comparing scores between architectures, only compare within the same dataset, so only compare DF2K scores with DF2K scores or DIV2K scores with DIV2K scores. DF2K scores are typically higher than DIV2K scores on the same architecture. PSNR and SSIM are calculated on the Y channel of the Urban100 validation dataset, one of the standard research validation sets.
""")

        for use_amp in [True]:
            for scale in ALL_SCALES:
                results_by_scale[scale] = []

                for name, arch, extra_arch_params in FILTERED_REGISTRIES_PARAMS:
                    arch_key = f"{name} {format_extra_params(extra_arch_params)}"
                    dtype_str, dtype = get_dtype(name, use_amp)
                    try:
                        # if "rcan" not in name:
                        #     continue
                        if arch_key not in results_by_arch:
                            results_by_arch[arch_key] = {}
                        row = benchmark_arch(
                            name, arch, extra_arch_params, torch.preserve_format
                        )
                        row_channels_last = benchmark_arch(
                            name, arch, extra_arch_params, torch.channels_last
                        )

                        channels_last_improvement = row_channels_last[3] / row[3]
                        new_row = (
                            row[0],
                            row[1],
                            row[2],
                            row[3],
                            row_channels_last[4],
                            row[5],
                            row[6],
                            row[7],
                            row_channels_last[3],
                            channels_last_improvement,
                        )
                        results_by_scale[scale].append(new_row)
                        results_by_arch[arch_key][scale] = new_row
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
                            float("inf"),
                            float("inf"),
                        )
                        results_by_scale[scale].append(row)
                        results_by_arch[arch_key][scale] = row
                    print(get_line(*results_by_scale[scale][-1]))

                results_by_scale[scale].sort(key=lambda x: x[8])

        printfc("## By Scale", f)

        for scale in ALL_SCALES:
            if print_markdown:
                f.write(f"\n### {scale}x scale\n")
                f.write(
                    f"{input_shape} input, {warmup_runs} warmup + {num_runs} runs averaged\n"
                )
                f.write(
                    "|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|\n"
                )
                f.write("|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|\n")
            for row in results_by_scale[scale]:
                printfc(get_line(*row, print_markdown), f)

        if print_markdown:
            f.write("\n## By Architecture\n")

        for arch_name in sorted(results_by_arch.keys()):
            f.write(f"\n### {arch_name}\n")
            runs = lightweight_num_runs if arch_name in LIGHTWEIGHT_ARCHS else num_runs
            f.write(
                f"{input_shape} input, {warmup_runs} warmup + {runs} runs averaged\n"
            )
            f.write(
                "|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|\n"
            )
            f.write("|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|\n")
            for scale in ALL_SCALES:
                f.write(
                    f"{get_line(*results_by_arch[arch_name][scale], print_markdown)}\n"
                )

        end_script_time = time.time()
        print(f"\nFinished in: {end_script_time - start_script_time:.2f} seconds")
