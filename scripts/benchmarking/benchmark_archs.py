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


def output_line(
    name: str,
    avg_time: float,
    fps: float,
    vram: float,
    params: int,
    scale: int,
    print_markdown: bool = False,
) -> None:
    name_separator = "|" if print_markdown else ": "
    separator = "|" if print_markdown else ",    "
    edge_separator = "|" if print_markdown else ""
    extra_edge_separator = "|||" if print_markdown else ""

    fps_label = "" if print_markdown else "FPS: "
    sec_img_label = "" if print_markdown else "sec/img: "
    vram_label = "" if print_markdown else "VRAM: "
    params_label = "" if print_markdown else "Params: "

    if params != -1:
        print(
            f"{edge_separator}{name:<18}{name_separator}{fps_label}{fps:>8.2f}{separator}{sec_img_label}{avg_time:>8.4f}{separator}{vram_label}{vram:>8.2f} GB{separator}{params_label}{params:>10,d}{edge_separator}"
        )
    else:
        print(
            f"{edge_separator}{name:<18}{name_separator}Unsupported at {scale}x{edge_separator}{extra_edge_separator}"
        )


def benchmark_model(
    model: nn.Module, input_tensor: Tensor, warmup_runs: int = 5, num_runs: int = 10
) -> tuple[float, Tensor]:
    for _ in range(warmup_runs):
        with torch.no_grad():
            model(input_tensor)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            output = model(input_tensor)
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time, output


if __name__ == "__main__":
    start_script_time = time.time()
    device = "cuda"

    input_shape = (1, 3, 480, 640)
    random_input = torch.rand(input_shape, device=device)
    n, c, h, w = input_shape
    scales = [4, 3, 2, 1]
    warmup_runs = 1
    num_runs = 10
    print_markdown = True
    results: dict[int, list[tuple[str, float, float, float, int, int]]] = {}

    for scale in scales:
        results[scale] = []

        for name, arch in SPANDREL_REGISTRY:
            if name in EXCLUDE_BENCHMARK_ARCHS:
                continue

            try:
                model = arch(scale=scale).eval().to(device)
                total_params = sum(p[1].numel() for p in model.named_parameters())

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                avg_time, output = benchmark_model(
                    model, random_input, warmup_runs, num_runs
                )

                if not (
                    output.shape[2] == random_input.shape[2] * scale
                    and output.shape[3] == random_input.shape[3] * scale
                ):
                    msg = f"{name}: {output.shape} is not {scale}x {random_input.shape}"
                    print(msg)
                    raise ValueError(msg)

                vram_usage = torch.cuda.max_memory_allocated(device) / (1024**3)
                results[scale].append(
                    (name, avg_time, 1 / avg_time, vram_usage, total_params, scale)
                )
            except ValueError:
                results[scale].append(
                    (
                        name,
                        float("inf"),
                        float("inf"),
                        float("inf"),
                        -1,
                        scale,
                    )
                )
            output_line(*results[scale][-1])

        results[scale].sort(key=lambda x: x[1])

    for scale in scales:
        print(
            f"\n{w}x{h} {c} channel input, {scale}x scale, {warmup_runs} warmup + {num_runs} runs averaged"
        )
        if print_markdown:
            print("|Name|FPS|sec/img|VRAM|Params|")
            print("|:-|-:|-:|-:|-:|")
        for row in results[scale]:
            output_line(*row, print_markdown)

    end_script_time = time.time()
    print(f"\nFinished in: {end_script_time - start_script_time:.2f} seconds")
