from __future__ import annotations

import torch
from torch import Tensor, nn

from traiNNer.data.degradations import resize_pt

_ALPHA_EPS = 1e-4


def compute_alpha(current_iter: int, end_iter: int) -> float:
    """Linear ramp 0 -> 1 over [0, end_iter]; paper says schedule shape is insignificant."""
    if end_iter <= 0:
        return 1.0
    return min(1.0, max(0.0, current_iter / end_iter))


def eco_synthesize(
    teacher: nn.Module, hr: Tensor, alpha: float, scale: int
) -> tuple[Tensor, Tensor]:
    """ECO (Eq. 14) mixup synthesis on HR, all OTF.

    Returns (input, target) where target = lerp(teacher(↓HR), HR, alpha) and
    input = ↓target. Antialiased bicubic is used for all downsamples.

    Mathematically equivalent (by linearity of bicubic) to the paper's two-lerp
    form: lerp(↓teacher_sr, LR, alpha) and lerp(teacher_sr, HR, alpha). The
    single-lerp form here costs one fewer lerp while producing identical tensors
    up to interpolation rounding.
    """
    if alpha >= 1.0 - _ALPHA_EPS:
        lr = resize_pt(hr, mode="bicubic", scale_factor=1.0 / scale)
        return lr, hr

    lr = resize_pt(hr, mode="bicubic", scale_factor=1.0 / scale)
    teacher_sr = teacher(lr).clamp_(0, 1)
    assert teacher_sr.shape == hr.shape, (
        f"ECO: teacher output shape {tuple(teacher_sr.shape)} != HR shape "
        f"{tuple(hr.shape)}; teacher scale/channels likely mismatched"
    )

    if alpha <= _ALPHA_EPS:
        target = teacher_sr
    else:
        target = torch.lerp(teacher_sr, hr, alpha)

    input_ = resize_pt(target, mode="bicubic", scale_factor=1.0 / scale)
    return input_, target
