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
    teacher: nn.Module, lq_real: Tensor, hr: Tensor, alpha: float, scale: int
) -> tuple[Tensor, Tensor]:
    """ECO mixup (paper Eq. 14, two-lerp form).

    Returns ``(lq_mix, gt_mix)`` where::

        teacher_sr = teacher(lq_real).clamp(0, 1)
        teacher_lq = bicubic(teacher_sr, 1/scale)
        gt_mix     = lerp(teacher_sr, hr,      alpha)
        lq_mix     = lerp(teacher_lq, lq_real, alpha)

    Fast paths: ``alpha >= 1`` returns ``(lq_real, hr)`` (teacher skipped);
    ``alpha <= 0`` returns ``(teacher_lq, teacher_sr)`` (saves one lerp).

    Works model-agnostically: ``lq_real`` may be a real LR from disk, a bicubic
    downsample of HR, an OTF-degraded tensor, etc. The caller is responsible
    for the upstream choice.
    """
    if alpha >= 1.0 - _ALPHA_EPS:
        return lq_real, hr

    teacher_sr = teacher(lq_real).clamp_(0, 1)
    assert teacher_sr.shape == hr.shape, (
        f"ECO: teacher output shape {tuple(teacher_sr.shape)} != HR shape "
        f"{tuple(hr.shape)}; teacher scale/channels likely mismatched"
    )
    teacher_lq = resize_pt(teacher_sr, mode="bicubic", scale_factor=1.0 / scale)

    if alpha <= _ALPHA_EPS:
        return teacher_lq, teacher_sr

    gt_mix = torch.lerp(teacher_sr, hr, alpha)
    lq_mix = torch.lerp(teacher_lq, lq_real, alpha)
    return lq_mix, gt_mix
