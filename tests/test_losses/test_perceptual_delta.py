import copy
from collections.abc import Callable

import pytest
import torch
from torch import Tensor
from traiNNer.losses import build_loss
from traiNNer.losses.perceptual_loss import PerceptualLoss


def _separate_passes(loss_fn: PerceptualLoss, x: Tensor, gt: Tensor) -> Tensor:
    # the formula the delta path replaces, VGG(x) and VGG(gt) run separately
    fx, fg = loss_fn.vgg(x), loss_fn.vgg(gt)
    total = torch.zeros((), dtype=x.dtype)
    for k in fx:
        d = fx[k] - fg[k]
        total = total + torch.sqrt(d * d + 1e-12).mean() * loss_fn.layer_weights[k]
    return total


@pytest.fixture(scope="module")
def loss_fn() -> PerceptualLoss:
    return PerceptualLoss(1.0)


@pytest.fixture(scope="module")
def images() -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(0)
    gt = torch.rand(1, 3, 32, 32, generator=g)
    x = (gt + 0.05 * torch.randn(gt.shape, generator=g)).clamp(0, 1)
    return x, gt


@pytest.fixture(scope="module")
def base(
    loss_fn: PerceptualLoss, images: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    return _loss_and_grad(loss_fn, *images)


def _loss_and_grad(
    fn: Callable[[Tensor, Tensor], Tensor], x: Tensor, gt: Tensor, upstream: float = 1.0
) -> tuple[Tensor, Tensor]:
    x = x.detach().clone().requires_grad_(True)
    loss = fn(x, gt)
    (loss * upstream).backward()
    assert x.grad is not None
    return loss.detach().double(), x.grad.double()


def test_matches_separate_passes_in_float64(
    loss_fn: PerceptualLoss,
    images: tuple[Tensor, Tensor],
    base: tuple[Tensor, Tensor],
) -> None:
    x, gt = images
    ref = copy.deepcopy(loss_fn).double()
    loss64, grad64 = _loss_and_grad(
        lambda a, b: _separate_passes(ref, a, b), x.double(), gt.double()
    )
    loss, grad = base
    assert abs(loss - loss64) / loss64 < 1e-3
    assert (grad - grad64).norm() / grad64.norm() < 0.3


@pytest.mark.parametrize("upstream", [2.0**-40, 2.0**14])
def test_gradient_does_not_underflow_or_overflow(
    loss_fn: PerceptualLoss,
    images: tuple[Tensor, Tensor],
    base: tuple[Tensor, Tensor],
    upstream: float,
) -> None:
    # Loss weights, GradScaler and batch size scale the gradient. The fp16 backward
    # picks its own power-of-2 scale, so a power-of-2 upstream changes nothing else.
    _, grad = base
    _, scaled = _loss_and_grad(loss_fn, *images, upstream)
    assert torch.isfinite(scaled).all()
    assert torch.equal(scaled / upstream, grad)


def test_same_under_autocast(
    loss_fn: PerceptualLoss,
    images: tuple[Tensor, Tensor],
    base: tuple[Tensor, Tensor],
) -> None:
    loss, _ = base
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss_amp = loss_fn(*images)
    assert loss_amp.dtype == torch.float32
    assert torch.allclose(loss_amp.double(), loss)


def test_fp16_alias_is_perceptual_loss(
    loss_fn: PerceptualLoss,
    images: tuple[Tensor, Tensor],
    base: tuple[Tensor, Tensor],
) -> None:
    alias = build_loss({"type": "perceptualfp16loss", "loss_weight": 1.0})
    assert isinstance(alias, PerceptualLoss)
    assert torch.equal(alias(*images).double(), base[0])
