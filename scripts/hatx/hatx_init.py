# ruff: noqa
# type: ignore
"""Build a HAT-X starting checkpoint from a HAT-L aligned one, and/or score a checkpoint on Urban100.

  python scripts/hatx/hatx_init.py VARIANT --src HAT_L_ALIGNED.safetensors --out INIT.safetensors
  python scripts/hatx/hatx_init.py VARIANT --ckpt CKPT.safetensors --eval datasets/benchmark/Urban100

Everything HAT-L aligned has under the same name and shape is copied. A HAB's bias table keeps its
trained values, stretched (bicubic, Swin V2 style) to a bigger window; the channel path's temperatures
start at 1 and its gate at 0; a dropped CAB's weights are left out. Eval: Urban100 x4 Y PSNR, 4-pixel crop.
"""

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from traiNNer.archs.hat_arch import hat_l_aligned
from traiNNer.archs.hatx_arch import hatx_l

VARIANTS = {
    "A": None,  # plain hat_l_aligned
    "B": {"big_blocks": (2, 5)},
    "Bq": {"big_blocks": (1, 2, 4, 5)},  # 4 of 6 per group at 32x32
    "Ball": {"big_blocks": (0, 1, 2, 3, 4, 5)},  # every HAB at 32x32
    "Ballc": {
        "big_blocks": (0, 1, 2, 3, 4, 5),
        "channel": True,
    },  # Ball + the channel path
    "C1": {"channel": True},
    "C2": {"channel": True, "use_cab": False, "cab_fade": 10000},
    "D": {"big_blocks": (2, 5), "channel": True},
    "E": {"big_blocks": (2, 5), "channel": True, "use_cab": False, "cab_fade": 10000},
}


def build(variant):
    kw = VARIANTS[variant]
    return hat_l_aligned(scale=4) if kw is None else hatx_l(scale=4, **kw)


def convert(model, src):
    sd = model.state_dict()
    out, notes = {}, []
    for k, v in sd.items():
        s = src.get(k)
        if s is not None and s.shape == v.shape:
            out[k] = s
        elif s is not None and k.endswith("attn.relative_position_bias_table"):
            old = math.isqrt(s.shape[0])
            t = s.float().view(old, old, -1)
            n = math.isqrt(v.shape[0])
            if n != old:
                t = F.interpolate(
                    t.permute(2, 0, 1)[None],
                    size=(n, n),
                    mode="bicubic",
                    align_corners=True,
                )[0].permute(1, 2, 0)
            out[k] = t.reshape(n * n, -1).contiguous().to(s.dtype)
        else:
            out[k] = v.to(next(iter(src.values())).dtype)
            notes.append(f"new: {k} {tuple(v.shape)}")
    unused = [k for k in src if k not in sd]
    return out, notes, unused


def y_psnr(a, b, crop=4):
    def y(x):
        x = x.astype(np.float64) / 255.0
        return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]

    a, b = y(a)[crop:-crop, crop:-crop], y(b)[crop:-crop, crop:-crop]
    return 10 * math.log10(255.0**2 / ((a - b) ** 2).mean())


@torch.no_grad()
def evaluate(model, root):
    model = model.cuda().eval()
    ps = []
    for h in sorted(glob.glob(os.path.join(root, "HR", "*.png"))):
        base = os.path.splitext(os.path.basename(h))[0]
        c = glob.glob(os.path.join(root, "LR_bicubic", "X4", base + "*"))
        if not c:
            continue
        lr = (
            torch.from_numpy(
                np.asarray(Image.open(c[0]).convert("RGB"), dtype=np.float32) / 255
            )
            .permute(2, 0, 1)[None]
            .cuda()
        )
        hr = np.asarray(Image.open(h).convert("RGB"))
        sr = model(lr).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        hh, ww = min(hr.shape[0], sr.shape[0]), min(hr.shape[1], sr.shape[1])
        ps.append(y_psnr((sr * 255.0).round().astype(np.uint8)[:hh, :ww], hr[:hh, :ww]))
    return sum(ps) / len(ps), len(ps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(VARIANTS))
    p.add_argument("--src")
    p.add_argument("--out")
    p.add_argument("--ckpt")
    p.add_argument("--eval")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build(a.variant)
    if a.src:
        sd, notes, unused = convert(model, load_file(a.src))
        for n in notes:
            print(n)
        print(
            f"{len(unused)} source tensors unused"
            + (f", e.g. {unused[0]}" if unused else "")
        )
        if a.out:
            save_file(sd, a.out)
            print("wrote", a.out)
    else:
        sd = load_file(a.ckpt)
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    print(
        f"{a.variant}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params"
    )
    if a.eval:
        psnr, n = evaluate(model, a.eval)
        print(f"{a.variant}: Urban100 x4 Y PSNR {psnr:.3f} dB ({n} images)", flush=True)


if __name__ == "__main__":
    main()
