import os
import sys
import time

import torch
from torch import Tensor, nn

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r"..\.."))
)
from traiNNer.archs import SPANDREL_REGISTRY

EXCLUDE_BENCHMARK_ARCHS = {"dat", "hat", "swinir"}
LIGHTWEIGHT_ARCHS = {
    "realcugan",
    "span",
    "compact",
    "ultracompact",
    "superultracompact",
}


def get_line(
    name: str,
    avg_time: float,
    fps: float,
    vram: float,
    params: int,
    scale: int,
    print_markdown: bool = False,
) -> str:
    name_separator = "|" if print_markdown else ": "
    separator = "|" if print_markdown else ",    "
    edge_separator = "|" if print_markdown else ""
    unsupported_value = "-"
    name_str = f"{name} {scale}x"

    fps_label = "" if print_markdown else "FPS: "
    sec_img_label = "" if print_markdown else "sec/img: "
    vram_label = "" if print_markdown else "VRAM: "
    params_label = "" if print_markdown else "Params: "

    if params != -1:
        return f"{edge_separator}{name_str:<21}{name_separator}{fps_label}{fps:>8.2f}{separator}{sec_img_label}{avg_time:>8.4f}{separator}{vram_label}{vram:>8.2f} GB{separator}{params_label}{params:>10,d}{edge_separator}"

    return f"{edge_separator}{name_str:<21}{name_separator}{fps_label}{unsupported_value:<8}{separator}{sec_img_label}{unsupported_value:<8}{separator}{vram_label}{unsupported_value:<8}{separator}{params_label}{unsupported_value:<10}{edge_separator}"


def benchmark_model(
    model: nn.Module, input_tensor: Tensor, warmup_runs: int = 5, num_runs: int = 10
) -> tuple[float, Tensor]:
    for _ in range(warmup_runs):
        with torch.inference_mode():
            model(input_tensor)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.inference_mode():
            output = model(input_tensor)
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time, output


if __name__ == "__main__":
    start_script_time = time.time()
    device = "cuda"

    input_shape = (1, 3, 480, 640)
    scales = [4, 3, 2, 1]
    warmup_runs = 1
    num_runs = 5
    lightweight_num_runs = 250
    use_fp16 = False
    print_markdown = True

    dtype_str = "fp16" if use_fp16 else "fp32"
    dtype = torch.float16 if use_fp16 else torch.float32
    random_input = torch.rand(input_shape, device=device, dtype=dtype)
    n, c, h, w = input_shape
    results_by_scale: dict[int, list[tuple[str, float, float, float, int, int]]] = {}
    results_by_arch: dict[
        str, dict[int, tuple[str, float, float, float, int, int]]
    ] = {}

    for scale in scales:
        results_by_scale[scale] = []

        for name, arch in SPANDREL_REGISTRY:
            if name in EXCLUDE_BENCHMARK_ARCHS:
                continue

            try:
                model = arch(scale=scale).eval().to(device, dtype=dtype)
                total_params = sum(p[1].numel() for p in model.named_parameters())
                runs = lightweight_num_runs if name in LIGHTWEIGHT_ARCHS else num_runs
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

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
                row = (name, avg_time, 1 / avg_time, vram_usage, total_params, scale)

                results_by_scale[scale].append(row)

                if name not in results_by_arch:
                    results_by_arch[name] = {}
                results_by_arch[name][scale] = row
            except ValueError:
                row = (
                    name,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    -1,
                    scale,
                )
                results_by_scale[scale].append(row)
                results_by_arch[name][scale] = row
            print(get_line(*results_by_scale[scale][-1]))

        results_by_scale[scale].sort(key=lambda x: x[1])

    if not print_markdown:
        print(
            f"\n{w}x{h} {c} channel input, {scale}x scale, {dtype_str}, {warmup_runs} warmup + {num_runs} ({lightweight_num_runs} for lightweight) runs averaged"
        )

    for scale in scales:
        if print_markdown:
            print(f"\n### {scale}x scale")
            print(
                f"{w}x{h} {c} channel input, {dtype_str}, {warmup_runs} warmup + {num_runs} runs averaged"
            )
            print("|Name|FPS|sec/img|VRAM|Params|")
            print("|:-|-:|-:|-:|-:|")
        for row in results_by_scale[scale]:
            print(get_line(*row, print_markdown))

    for arch_name in sorted(results_by_arch.keys()):
        print(f"\n### {arch_name}")
        runs = lightweight_num_runs if arch_name in LIGHTWEIGHT_ARCHS else num_runs
        print(
            f"{w}x{h} {c} channel input, {dtype_str}, {warmup_runs} warmup + {runs} runs averaged"
        )
        print("|Name|FPS|sec/img|VRAM|Params|")
        print("|:-|-:|-:|-:|-:|")
        for scale in scales:
            print(get_line(*results_by_arch[arch_name][scale], print_markdown))

    end_script_time = time.time()
    print(f"\nFinished in: {end_script_time - start_script_time:.2f} seconds")
