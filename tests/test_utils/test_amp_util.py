"""A non-finite gradient must skip the optimizer step, with and without loss scaling."""

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from traiNNer.utils.amp_util import (
    grad_scaler_scale,
    guard_nonfinite,
    make_grad_scaler,
    update_grad_scaler,
)

CASES = [("cpu", False)] + (
    [("cuda", False), ("cuda", True)] if torch.cuda.is_available() else []
)


@pytest.mark.parametrize(("device", "fused"), CASES)
@pytest.mark.parametrize("loss_scaling", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_loss_skips_step(
    device: str, fused: bool, loss_scaling: bool, bad: float
) -> None:
    if loss_scaling and device == "cpu":
        pytest.skip("fp16 loss scaling is a CUDA path")
    torch.manual_seed(0)
    net = nn.Linear(4, 4).to(device)
    opt = AdamW(net.parameters(), lr=1e-2, fused=fused)
    scaler = make_grad_scaler(loss_scaling, device)
    x = torch.rand(8, 4, device=device)

    def step(loss_offset: float) -> None:
        # a constant offset has no gradient path, so only the guard makes the
        # grads non-finite
        loss, _ = guard_nonfinite(net(x).square().mean() + loss_offset)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        update_grad_scaler(scaler, loss_scaling)
        opt.zero_grad()

    step(0.0)  # a normal step, so the optimizer has state
    before = [p.detach().clone() for p in net.parameters()]
    scale_before = float(grad_scaler_scale(scaler))
    step(bad)
    for p, b in zip(net.parameters(), before, strict=True):
        torch.testing.assert_close(p.detach(), b, rtol=0, atol=0)

    scale_after = float(grad_scaler_scale(scaler))
    if loss_scaling:
        assert scale_after < scale_before  # backed off
    else:
        assert scale_after == 1.0  # pinned

    step(0.0)  # training resumes
    assert any(
        not torch.equal(p.detach(), b)
        for p, b in zip(net.parameters(), before, strict=True)
    )
