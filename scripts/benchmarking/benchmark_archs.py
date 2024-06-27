import os
import sys
import time

import torch
from torch import Tensor, nn

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r"..\.."))
)
from traiNNer.archs import SPANDREL_REGISTRY

BENCHMARK_ARCHS = [
    "atd_light",
    "atd",
    "compact",
    "dat_2",
    "esrgan_lite",
    "esrgan",
    "hat_l",
    "hat_m",
    "hat_s",
    "omnisr",
    "plksr",
    "realcugan",
    "realplksr",
    "span",
    "srformer_light",
    "srformer",
    "superultracompact",
    "swinir_l",
    "swinir_m",
    "swinir_s",
    "ultracompact",
]


def benchmark_model(
    model: nn.Module, input_tensor: Tensor, num_runs: int = 10
) -> tuple[float, Tensor]:
    # warm up
    for _ in range(5):
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
    device = "cuda"

    input_shape = (1, 3, 480, 640)
    random_input = torch.rand(input_shape, device=device)
    n, c, h, w = random_input.shape
    scale = 2
    num_runs = 10

    print(f"upscaling {w}x{h} input by {scale}x over {num_runs} runs")

    for name in BENCHMARK_ARCHS:
        arch = SPANDREL_REGISTRY.get(name)
        model = arch(scale=scale).eval().to(device)

        avg_time, output = benchmark_model(model, random_input, num_runs)

        assert (
            output.shape[2] == random_input.shape[2] * scale
            and output.shape[3] == random_input.shape[3] * scale
        )

        print(f"{name}: {1 / avg_time:.2f} fps ({avg_time:.4f} seconds per image)")
