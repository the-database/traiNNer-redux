import pytest
import torch
from torch import Tensor, nn
from traiNNer.utils.eco import compute_alpha, eco_synthesize


class IdentityUpTeacher(nn.Module):
    """Teacher = bicubic upsample. Used for the linearity/equivalence tests."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, lr: Tensor) -> Tensor:
        return torch.nn.functional.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", antialias=True
        )


class ExplodingTeacher(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        raise AssertionError("teacher should not be called on fast path")


def test_compute_alpha_linear_schedule() -> None:
    assert compute_alpha(0, 100) == 0.0
    assert compute_alpha(25, 100) == 0.25
    assert compute_alpha(50, 100) == 0.5
    assert compute_alpha(100, 100) == 1.0
    assert compute_alpha(200, 100) == 1.0  # clamped past end
    # end_iter == 0 edge case
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
    teacher = ExplodingTeacher()
    lq, gt = eco_synthesize(teacher, hr, alpha=1.0, scale=4)
    assert lq.shape == (2, 3, 16, 16)
    assert gt.shape == (2, 3, 64, 64)
    assert torch.equal(gt, hr)


def test_eco_synthesize_shapes() -> None:
    hr = torch.rand(2, 3, 128, 128)
    teacher = IdentityUpTeacher(scale=4)
    lq, gt = eco_synthesize(teacher, hr, alpha=0.5, scale=4)
    assert lq.shape == (2, 3, 32, 32)
    assert gt.shape == (2, 3, 128, 128)


def test_eco_synthesize_alpha_zero_target_is_teacher_output() -> None:
    # At alpha=0, target must equal teacher(↓HR) exactly (no lerp).
    hr = torch.rand(1, 3, 64, 64)
    teacher = IdentityUpTeacher(scale=4)
    _, gt = eco_synthesize(teacher, hr, alpha=0.0, scale=4)
    expected_lr = torch.nn.functional.interpolate(
        hr, scale_factor=0.25, mode="bicubic", antialias=True
    )
    expected_teacher = teacher(expected_lr).clamp_(0, 1)
    assert torch.allclose(gt, expected_teacher, atol=1e-6)


def test_eco_synthesize_spatial_consistency() -> None:
    # By construction: input = ↓target (with antialiased bicubic).
    hr = torch.rand(1, 3, 64, 64)
    teacher = IdentityUpTeacher(scale=4)
    for alpha in (0.0, 0.25, 0.5, 0.75):
        lq, gt = eco_synthesize(teacher, hr, alpha=alpha, scale=4)
        ds_gt = torch.nn.functional.interpolate(
            gt, scale_factor=0.25, mode="bicubic", antialias=True
        )
        assert torch.allclose(lq, ds_gt, atol=1e-5), f"alpha={alpha}"


def test_eco_synthesize_teacher_shape_mismatch_raises() -> None:
    class WrongScaleTeacher(nn.Module):
        def forward(self, lr: Tensor) -> Tensor:
            # returns 2x instead of 4x
            return torch.nn.functional.interpolate(lr, scale_factor=2, mode="bicubic")

    hr = torch.rand(1, 3, 64, 64)
    with pytest.raises(AssertionError, match="teacher output shape"):
        eco_synthesize(WrongScaleTeacher(), hr, alpha=0.5, scale=4)
