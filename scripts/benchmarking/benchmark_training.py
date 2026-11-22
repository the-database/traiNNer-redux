"""Measure full train.py updates on paired images for a registered image SR generator."""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import yaml
from scripts.benchmarking.benchmark_paired_training import prepare_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", nargs="+", required=True, help="Generator registry names"
    )
    parser.add_argument(
        "--checkout", type=Path, default=Path.cwd(), help="Trainer checkout to execute"
    )
    parser.add_argument(
        "--paired-root",
        type=Path,
        help="Directory containing matching hr/ and lr/ images",
    )
    parser.add_argument(
        "--network-options",
        type=json.loads,
        default={},
        help="Additional network_g options as JSON",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--scale", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--lr-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-mib", type=int, default=0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Updates per window. Discard the first of five windows.",
    )
    args = parser.parse_args()
    if args.interval < 4 or min(args.batch_size, args.lr_size) < 1:
        parser.error(
            "use at least four updates per window and positive batch and crop sizes"
        )
    root = args.checkout.resolve()
    work = (Path("tmp") / f"train-arch-{uuid4().hex[:12]}").resolve()
    work.mkdir(parents=True)
    pairs = args.paired_root.resolve() if args.paired_root else work
    if args.paired_root is None:
        prepare_pairs(work, args.scale)
    config = yaml.safe_load(
        (root / "options/_templates/train/SPAN/SPAN_S_fidelity.yml").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        scale=args.scale,
        manual_seed=141,
        num_gpu=1,
        use_amp=True,
        amp_bf16=True,
        use_channels_last=True,
        use_cuda_graph=args.cuda_graph,
    )
    config["datasets"].pop("val")
    config["datasets"]["train"].update(
        dataroot_gt=[str(pairs / "hr")],
        dataroot_lq=[str(pairs / "lr")],
        num_worker_per_gpu=args.workers,
        batch_size_per_gpu=args.batch_size,
        lq_size=args.lr_size,
        accum_iter=1,
        prefetch_mode="cuda",
    )
    if args.cache_mib:
        config["datasets"]["train"]["decoded_cache_mib"] = args.cache_mib
    config["train"].update(total_iter=5 * args.interval, ema_decay=0)
    config["val"]["val_enabled"] = False
    config["logger"].update(
        print_freq=args.interval,
        save_checkpoint_freq=5 * args.interval,
        use_tb_logger=False,
    )
    print(f"Configs and logs: {work}", flush=True)
    failed = []
    for arch in args.arch:
        config["name"] = f"{work.name}-{arch}"
        config["network_g"] = {"type": arch, **args.network_options}
        path = work / f"{arch}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        console_path = work / f"{arch}.log"
        with console_path.open("w", encoding="utf-8") as console:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "train.py", "-opt", str(path)],
                cwd=root,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode:
            print(f"{arch}: exit {proc.returncode}, see {console_path}", flush=True)
            failed.append(arch)
            continue
        directory = root / "experiments" / config["name"]
        log = next(directory.glob("train_*.log")).read_text(encoding="utf-8")
        rates = [float(v) for v in re.findall(r"it/s, ([\d.]+) samples/s", log)]
        graph = "recorded the training step" in log and "CUDA graph disabled" not in log
        if (
            len(rates) != 5
            or "NaN/Inf in loss" in log
            or (args.cuda_graph and not graph)
        ):
            print(
                f"{arch}: missing timings or graph capture, see {console_path}",
                flush=True,
            )
            failed.append(arch)
            continue
        print(
            f"{arch}: {statistics.harmonic_mean(rates[1:]):.2f} crops/s | windows {rates[1:]}",
            flush=True,
        )
    if failed:
        raise SystemExit(f"Unmeasured: {', '.join(failed)}")


if __name__ == "__main__":
    main()
