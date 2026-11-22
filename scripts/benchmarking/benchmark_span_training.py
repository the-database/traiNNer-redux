"""Time SPANF3 forward, backward and AdamW on the same resident image pairs."""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F  # noqa: N812
from traiNNer.archs.spanf3_arch import spanf3
from traiNNer.utils.cuda_graph import CudaGraphStep


def measure(args: argparse.Namespace, fused: bool) -> tuple[float, float, float]:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    blocks, channels = (5, 48) if args.preset == "performance" else (8, 64)
    model = (
        spanf3(
            scale=2, feature_channels=channels, num_blocks=blocks, fused_training=fused
        )
        .cuda()
        .to(memory_format=torch.channels_last)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, fused=True, capturable=args.cuda_graph
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    graph = CudaGraphStep([optimizer], [scaler]) if args.cuda_graph else None
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    size = args.lr_size
    lr = torch.rand(args.batch_size, 3, size, size, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    hr = torch.rand(args.batch_size, 3, 2 * size, 2 * size, device="cuda")

    inputs = {"lr": lr, "hr": hr}

    def step(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype, enabled=args.precision != "fp32"):
            loss = F.l1_loss(model(batch["lr"]).float(), batch["hr"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=graph is None)
        return loss.detach()

    def update() -> torch.Tensor:
        return step(inputs) if graph is None else graph(True, inputs, step)

    for _ in range(args.warmup):
        update()
    if graph is not None and not graph.enabled:
        raise RuntimeError("CUDA graph capture failed")
    counter = optimizer.state[next(model.parameters())]["step"]
    before = int(counter.item())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = update()
    for _ in range(1, args.steps):
        loss = update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if int(counter.item()) - before != args.steps:
        raise RuntimeError("AdamW did not apply every timed update")
    if not loss.isfinite().item():
        raise RuntimeError("Non-finite training loss")
    return (
        args.batch_size * args.steps / elapsed,
        torch.cuda.max_memory_allocated() / 2**30,
        torch.cuda.max_memory_reserved() / 2**30,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=("performance", "balanced"), default="performance"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-size", type=int, default=64)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.lr_size, args.warmup, args.steps, args.rounds) < 1:
        parser.error("batch size, crop size, warmup, steps and rounds must be positive")
    if args.cuda_graph and args.warmup < 4:
        parser.error("CUDA graph capture needs at least four warmup steps")
    print(
        f"{torch.cuda.get_device_name()} | torch {torch.__version__} | {vars(args)}",
        flush=True,
    )
    torch.backends.cudnn.benchmark = True
    results: dict[bool, list[float]] = {False: [], True: []}
    for i in range(args.rounds):
        for fused in (True, False) if i % 2 else (False, True):
            speed, allocated, reserved = measure(args, fused)
            results[fused].append(speed)
            print(
                f"round {i + 1} | {'fused' if fused else 'branches'} | "
                f"{speed:.1f} LR crops/s | {allocated:.3f} GiB allocated, "
                f"{reserved:.3f} GiB reserved at peak",
                flush=True,
            )
    baseline, fused_rate = (statistics.median(results[k]) for k in (False, True))
    print(
        f"Median {baseline:.1f} -> {fused_rate:.1f} LR crops/s ({fused_rate / baseline:.2f}x)"
    )


if __name__ == "__main__":
    main()
