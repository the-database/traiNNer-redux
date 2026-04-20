from __future__ import annotations

from typing import Literal

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
    teacher: nn.Module,
    lq_real: Tensor,
    hr: Tensor,
    alpha: float,
    scale: int,
    mode: Literal["full", "hr_only"] = "full",
) -> tuple[Tensor, Tensor]:
    """ECO mixup.

    Returns ``(lq_mix, gt_mix)``.

    mode="full" (paper Eq. 14, two-lerp form)::

        teacher_sr = teacher(lq_real).clamp(0, 1)
        teacher_lq = bicubic(teacher_sr, 1/scale)
        gt_mix     = lerp(teacher_sr, hr,      alpha)
        lq_mix     = lerp(teacher_lq, lq_real, alpha)

    mode="hr_only" (scheduled KD form — input stays as real LR)::

        teacher_sr = teacher(lq_real).clamp(0, 1)
        gt_mix     = lerp(teacher_sr, hr, alpha)
        lq_mix     = lq_real

    In both modes ``alpha >= 1`` returns ``(lq_real, hr)`` (teacher skipped).
    In ``full`` ``alpha <= 0`` returns ``(teacher_lq, teacher_sr)``; in
    ``hr_only`` ``alpha <= 0`` returns ``(lq_real, teacher_sr)``.
    """
    if alpha >= 1.0 - _ALPHA_EPS:
        return lq_real, hr

    teacher_sr = teacher(lq_real).clamp_(0, 1)
    assert teacher_sr.shape == hr.shape, (
        f"ECO: teacher output shape {tuple(teacher_sr.shape)} != HR shape "
        f"{tuple(hr.shape)}; teacher scale/channels likely mismatched"
    )

    if mode == "hr_only":
        if alpha <= _ALPHA_EPS:
            return lq_real, teacher_sr
        return lq_real, torch.lerp(teacher_sr, hr, alpha)

    # mode == "full"
    teacher_lq = resize_pt(teacher_sr, mode="bicubic", scale_factor=1.0 / scale)
    if alpha <= _ALPHA_EPS:
        return teacher_lq, teacher_sr
    gt_mix = torch.lerp(teacher_sr, hr, alpha)
    lq_mix = torch.lerp(teacher_lq, lq_real, alpha)
    return lq_mix, gt_mix
