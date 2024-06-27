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
    n, c, h, w = random_input.shape
    scale = 2
    warmup_runs = 5
    num_runs = 10

    results = []

    for name in BENCHMARK_ARCHS:
        arch = SPANDREL_REGISTRY.get(name)
        model = arch(scale=scale).eval().to(device)

        avg_time, output = benchmark_model(model, random_input, warmup_runs, num_runs)

        assert (
            output.shape[2] == random_input.shape[2] * scale
            and output.shape[3] == random_input.shape[3] * scale
        )

        results.append((name, avg_time, 1 / avg_time))
        print(f"{name:<18}: {1 / avg_time:>7.2f} fps ({avg_time:.4f} seconds per image)")

    results.sort(key=lambda x: x[1])

    print(
        f"\n{w}x{h} {c} channel input, {scale}x scale, {warmup_runs} warmup + {num_runs} runs averaged"
    )
    for name, avg_time, fps in results:
        # print(f"{name}: {fps:.2f} fps ({avg_time:.4f} seconds per image)")
         print(f"{name:<18}: {fps:>7.2f} fps ({avg_time:.4f} seconds per image)")

    end_script_time = time.time()
    print(f"\nFinished in: {end_script_time - start_script_time:.2f} seconds")
