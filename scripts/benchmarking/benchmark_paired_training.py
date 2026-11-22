"""Run four real train.py jobs with matched 2x PNG pairs, settings and seeds."""

import argparse
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image


def prepare_pairs(work: Path) -> None:
    for directory in ("hr", "lr"):
        (work / directory).mkdir()
    for source in sorted(Path("datasets/train/dataset1/hr").glob("*.png")):
        with Image.open(source) as image:
            hr = image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
            hr.save(work / "hr" / source.name)
            hr.resize((256, 256), Image.Resampling.BICUBIC).save(
                work / "lr" / source.name
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-root",
        type=Path,
        help="2x PNG pairs in hr/ and lr/. Default uses the bundled images.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache-mib",
        type=int,
        default=32,
        help="Decoded image budget per worker in the cached runs.",
    )
    args = parser.parse_args()
    if args.workers < 0 or args.cache_mib < 1:
        parser.error("workers must be nonnegative and cache-mib must be positive")
    Path("tmp").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="paired-span-", dir="tmp")).resolve()
    pairs = args.paired_root
    if pairs is None:
        prepare_pairs(work)
        pairs = work
    template = Path("options/_templates/train/SPAN/SPAN_S_fidelity.yml")
    opt = yaml.safe_load(template.read_text(encoding="utf-8"))
    opt.update(scale=2, num_gpu=1, manual_seed=141, amp_bf16=True, use_cuda_graph=True)
    opt["datasets"].pop("val")
    data = opt["datasets"]["train"]
    data.update(
        dataroot_gt=[str(pairs / "hr")],
        dataroot_lq=[str(pairs / "lr")],
        num_worker_per_gpu=args.workers,
        batch_size_per_gpu=16,
        accum_iter=1,
        dataset_enlarge_ratio=500 if args.paired_root is None else "auto",
        lq_size=96,
        prefetch_mode="cuda",
    )
    opt["train"].update(total_iter=100, ema_decay=0)
    opt["logger"].update(print_freq=20, save_checkpoint_freq=1000, use_tb_logger=False)
    opt["val"]["val_enabled"] = False
    print(f"Configs and console logs in {work}", flush=True)
    for fused, cache in (
        (False, 0),
        (True, 0),
        (True, args.cache_mib),
        (False, args.cache_mib),
    ):
        name = f"{'fused' if fused else 'branches'}-cache{cache}"
        opt["name"] = f"{work.name}-{name}"
        opt["network_g"] = {
            "type": "spanf3",
            "feature_channels": 64,
            "num_blocks": 8,
            "fused_training": fused,
        }
        data["decoded_cache_mib"] = cache
        config = work / f"{name}.yml"
        config.write_text(yaml.safe_dump(opt, sort_keys=False), encoding="utf-8")
        with (work / f"{name}.log").open("w", encoding="utf-8") as console:
            subprocess.run(
                [sys.executable, "-X", "utf8", "train.py", "-opt", str(config)],
                stdout=console,
                stderr=subprocess.STDOUT,
                check=True,
            )
        log = next((Path("experiments") / opt["name"]).glob("train_*.log")).read_text(
            encoding="utf-8"
        )
        rates = [float(v) for v in re.findall(r"it/s, ([\d.]+) samples/s", log)]
        if len(rates) != 5 or "recorded the training step" not in log:
            raise RuntimeError(f"Missing timings or CUDA graph in {name}")
        # The last four intervals have equal sample counts. Their harmonic mean
        # is total samples divided by total elapsed time after warmup.
        print(f"{name} | {statistics.harmonic_mean(rates[1:]):.1f} crops/s", flush=True)


if __name__ == "__main__":
    main()
