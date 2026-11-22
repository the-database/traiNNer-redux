"""Hold out validation pairs from a symlinked paired dataset, by source image.

  python scripts/hatx/split_pairs.py SRC_DIR OUT_TRAIN OUT_VAL [--n 100] [--lq 256] [--seed 0]

SRC_DIR holds hr/ and lr/ with matching file names (symlinks). Pairs whose hr resolves to the same
file are copies of one source with different degradations, so a held-out source takes all its copies
out of training. OUT_TRAIN gets hr/ and lr/ symlinks to the resolved files for the rest. OUT_VAL gets
one pair per held-out source, centre-cropped to an lq x lq LR patch (and the matching HR) as PNGs.
"""
import argparse, os, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("out_train")
    p.add_argument("out_val")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--lq", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    names = sorted(os.listdir(os.path.join(a.src, "lr")))
    with ThreadPoolExecutor(32) as ex:
        hr = list(ex.map(lambda n: os.path.realpath(os.path.join(a.src, "hr", n)), names, chunksize=2048))
        lr = list(ex.map(lambda n: os.path.realpath(os.path.join(a.src, "lr", n)), names, chunksize=2048))
    groups = defaultdict(list)
    for i, h in enumerate(hr):
        groups[h].append(i)
    rng = random.Random(a.seed)
    keys = sorted(groups)
    rng.shuffle(keys)

    os.makedirs(os.path.join(a.out_val, "hr"), exist_ok=True)
    os.makedirs(os.path.join(a.out_val, "lr"), exist_ok=True)
    held, val = set(), 0
    for k in keys:
        if val == a.n:
            break
        i = rng.choice(groups[k])
        lq = Image.open(lr[i]).convert("RGB")
        gt = Image.open(hr[i]).convert("RGB")
        s = gt.width // lq.width
        if gt.width != lq.width * s or gt.height != lq.height * s or min(lq.size) < a.lq:
            continue
        x, y = (lq.width - a.lq) // 2, (lq.height - a.lq) // 2
        base = f"{val:03d}.png"
        lq.crop((x, y, x + a.lq, y + a.lq)).save(os.path.join(a.out_val, "lr", base))
        gt.crop((x * s, y * s, (x + a.lq) * s, (y + a.lq) * s)).save(os.path.join(a.out_val, "hr", base))
        with open(os.path.join(a.out_val, "sources.txt"), "a") as f:
            f.write(f"{base}\t{names[i]}\n")
        held.add(k)
        val += 1

    for d in ("hr", "lr"):
        os.makedirs(os.path.join(a.out_train, d), exist_ok=True)
    kept = 0
    for i, n in enumerate(names):
        if hr[i] in held:
            continue
        os.symlink(hr[i], os.path.join(a.out_train, "hr", n))
        os.symlink(lr[i], os.path.join(a.out_train, "lr", n))
        kept += 1
    print(f"{len(names)} pairs, {len(groups)} sources; held out {len(held)} sources "
          f"({len(names) - kept} pairs), {val} val crops; {kept} training pairs")


if __name__ == "__main__":
    main()
