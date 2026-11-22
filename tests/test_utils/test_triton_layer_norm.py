"""The Triton LayerNorm must match nn.LayerNorm in values, dtypes and autograd."""

import importlib.util
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs CUDA and triton",
)

CASES = [
    # shape, input dtype, autocast dtype
    ((8, 4096, 180), torch.float32, torch.bfloat16),
    ((8, 4096, 180), torch.bfloat16, torch.bfloat16),
    ((8, 4096, 180), torch.float32, None),
    ((16, 1024, 96), torch.float32, torch.float16),
    ((3, 100, 60), torch.float32, None),  # rows not a multiple of the tile
    ((2, 7, 1024), torch.float32, torch.bfloat16),
]


def _run(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    amp: torch.dtype | None,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    x = x0.clone().requires_grad_(True)
    ww = w.clone().requires_grad_(True)
    bb = b.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
        y = fn(x, ww, bb)
    y.backward(dy.to(y.dtype))
    assert x.grad is not None and ww.grad is not None and bb.grad is not None
    return y, x.grad, ww.grad, bb.grad


@pytest.mark.parametrize(("shape", "x_dtype", "amp"), CASES)
def test_matches_torch(
    shape: tuple[int, ...], x_dtype: torch.dtype, amp: torch.dtype | None
) -> None:
    from traiNNer.ops.layer_norm import layer_norm

    torch.manual_seed(0)
    n = shape[-1]
    x0 = (torch.randn(shape, device="cuda") * 3 + 1).to(x_dtype)
    w = torch.randn(n, device="cuda") * 0.5 + 1
    b = torch.randn(n, device="cuda") * 0.1
    dy = torch.randn(shape, device="cuda")
    ref = _run(lambda x, ww, bb: F.layer_norm(x, (n,), ww, bb, 1e-5), x0, w, b, amp, dy)
    new = _run(lambda x, ww, bb: layer_norm(x, ww, bb, 1e-5), x0, w, b, amp, dy)
    for r, c in zip(ref, new, strict=True):
        assert r.dtype == c.dtype
        assert r.shape == c.shape
        tol = 2e-2 if r.dtype == torch.bfloat16 else 2e-4
        torch.testing.assert_close(c.float(), r.float(), rtol=tol, atol=tol)


def test_module_swap_keeps_state_dict_and_allows_inplace() -> None:
    from traiNNer.ops.layer_norm import TritonLayerNorm, use_triton_layer_norm

    net = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.ReLU()).cuda()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    assert use_triton_layer_norm(net) == 1
    assert isinstance(net[1], TritonLayerNorm)
    assert isinstance(net[1], nn.LayerNorm)
    assert net.state_dict().keys() == before.keys()
    x = torch.randn(4, 16, device="cuda", requires_grad=True)
    y = net[1](net[0](x))
    y.add_(1)  # some archs modify the LayerNorm output in place
    y.sum().backward()
    assert x.grad is not None
    # still falls back on the CPU
    cpu = nn.LayerNorm(8)
    cpu.__class__ = TritonLayerNorm
    torch.testing.assert_close(
        cpu(torch.ones(2, 8)), F.layer_norm(torch.ones(2, 8), (8,))
    )
