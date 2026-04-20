import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from traiNNer.utils.eco import compute_alpha, eco_synthesize


class IdentityUpTeacher(nn.Module):
    """Teacher = bicubic upsample. Used for the linearity/equivalence tests."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, lr: Tensor) -> Tensor:
        return F.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", antialias=True
        )


class ExplodingTeacher(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        raise AssertionError("teacher should not be called on fast path")


def _bicubic_ds(t: Tensor, factor: float) -> Tensor:
    return F.interpolate(t, scale_factor=factor, mode="bicubic", antialias=True).clamp(
        0, 1
    )


def test_compute_alpha_linear_schedule() -> None:
    assert compute_alpha(0, 100) == 0.0
    assert compute_alpha(25, 100) == 0.25
    assert compute_alpha(50, 100) == 0.5
    assert compute_alpha(100, 100) == 1.0
    assert compute_alpha(200, 100) == 1.0  # clamped past end
    assert compute_alpha(0, 0) == 1.0


def test_compute_alpha_monotonic() -> None:
    prev = -1.0
    for i in range(101):
        a = compute_alpha(i, 100)
        assert a >= prev
        prev = a


def test_eco_synthesize_alpha_one_skips_teacher() -> None:
    # alpha=1.0 must hit the fast path and never invoke the teacher.
    hr = torch.rand(2, 3, 64, 64)
    lq = torch.rand(2, 3, 16, 16)
    teacher = ExplodingTeacher()
    lq_mix, gt_mix = eco_synthesize(teacher, lq, hr, alpha=1.0, scale=4)
    assert torch.equal(lq_mix, lq)
    assert torch.equal(gt_mix, hr)


def test_eco_synthesize_alpha_zero_returns_teacher_pair() -> None:
    # At alpha=0, target is teacher output and input is its bicubic downsample.
    hr = torch.rand(1, 3, 64, 64)
    lq = _bicubic_ds(hr, 0.25)
    teacher = IdentityUpTeacher(scale=4)
    lq_mix, gt_mix = eco_synthesize(teacher, lq, hr, alpha=0.0, scale=4)
    expected_teacher_sr = teacher(lq).clamp_(0, 1)
    expected_teacher_lq = _bicubic_ds(expected_teacher_sr, 0.25)
    assert torch.allclose(gt_mix, expected_teacher_sr, atol=1e-6)
    assert torch.allclose(lq_mix, expected_teacher_lq, atol=1e-6)


def test_eco_synthesize_shapes() -> None:
    hr = torch.rand(2, 3, 128, 128)
    lq = torch.rand(2, 3, 32, 32)
    teacher = IdentityUpTeacher(scale=4)
    lq_mix, gt_mix = eco_synthesize(teacher, lq, hr, alpha=0.5, scale=4)
    assert lq_mix.shape == (2, 3, 32, 32)
    assert gt_mix.shape == (2, 3, 128, 128)


def test_eco_synthesize_spatial_consistency_under_bicubic_linearity() -> None:
    # When lq_real = ↓hr, the two-lerp form equals single-lerp-then-downsample
    # by bicubic linearity. Specifically, ↓gt_mix ≈ lq_mix.
    hr = torch.rand(1, 3, 64, 64)
    lq = _bicubic_ds(hr, 0.25)
    teacher = IdentityUpTeacher(scale=4)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        lq_mix, gt_mix = eco_synthesize(teacher, lq, hr, alpha=alpha, scale=4)
        ds_gt = _bicubic_ds(gt_mix, 0.25)
        assert torch.allclose(lq_mix, ds_gt, atol=1e-5), f"alpha={alpha}"


def test_eco_synthesize_paired_real_lq_endpoints() -> None:
    # Paired mode: lq_real is NOT ↓hr. Endpoints must still be well-defined.
    hr = torch.rand(1, 3, 64, 64)
    lq_real = torch.rand(1, 3, 16, 16)  # independent, simulates custom degradation
    teacher = IdentityUpTeacher(scale=4)

    # alpha=1 returns the vanilla paired input/target.
    lq_mix, gt_mix = eco_synthesize(teacher, lq_real, hr, alpha=1.0, scale=4)
    assert torch.equal(lq_mix, lq_real)
    assert torch.equal(gt_mix, hr)

    # alpha=0 returns the teacher-anchored synthetic pair.
    lq_mix0, gt_mix0 = eco_synthesize(teacher, lq_real, hr, alpha=0.0, scale=4)
    expected_teacher_sr = teacher(lq_real).clamp_(0, 1)
    expected_teacher_lq = _bicubic_ds(expected_teacher_sr, 0.25)
    assert torch.allclose(gt_mix0, expected_teacher_sr, atol=1e-6)
    assert torch.allclose(lq_mix0, expected_teacher_lq, atol=1e-6)


def test_eco_synthesize_teacher_shape_mismatch_raises() -> None:
    class WrongScaleTeacher(nn.Module):
        def forward(self, lr: Tensor) -> Tensor:
            return F.interpolate(lr, scale_factor=2, mode="bicubic")  # wrong scale

    hr = torch.rand(1, 3, 64, 64)
    lq = torch.rand(1, 3, 16, 16)
    with pytest.raises(AssertionError, match="teacher output shape"):
        eco_synthesize(WrongScaleTeacher(), lq, hr, alpha=0.5, scale=4)
