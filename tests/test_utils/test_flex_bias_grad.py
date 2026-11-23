"""FlexBiasGrad must give FlexAttention's own bias table gradient, and leave q, k and v's alone."""

import pytest
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import AuxRequest, flex_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="needs a CUDA GPU (sm_80+)",
)


@torch.compile(dynamic=True)
def _flex(
    q: Tensor, k: Tensor, v: Tensor, table: Tensor, rpi: Tensor, region: Tensor
) -> tuple[Tensor, Tensor]:
    def score_mod(score, b, h, qi, ki):  # noqa: ANN001, ANN202
        w = b % region.shape[0]
        same = region[w, qi] == region[w, ki]
        return score + table[rpi[qi, ki], h] + torch.where(same, 0.0, -100.0)

    out, aux = flex_attention(
        q, k, v, score_mod=score_mod, scale=1.0, return_aux=AuxRequest(lse=True)
    )
    assert aux.lse is not None
    return out, aux.lse


def rel(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


@pytest.mark.parametrize(
    ("nq", "nk", "nreg"),
    [(256, 256, 4), (1024, 1024, 4), (256, 576, 1), (64, 144, 1), (100, 100, 2)],
)
def test_matches_flex_and_float64(nq: int, nk: int, nreg: int) -> None:
    """Window 16 and 32 with a shifted-window mask, HAT's OCA (16 x 16 queries on 24 x 24 keys),
    window 8 and a size that isn't a multiple of the tile."""
    from traiNNer.ops.flex_bias_grad import FlexBiasGrad

    torch.manual_seed(0)
    nw, nh, d, rows = 8, 3, 32, 500
    pad_q, pad_k = -nq % 128, -nk % 128
    rpi = torch.randint(
        0, rows, (nq + pad_q, nk + pad_k), device="cuda", dtype=torch.int32
    )
    region = torch.randint(0, 3, (nreg, max(nq + pad_q, nk + pad_k)), device="cuda")
    region = region.to(torch.int32)
    table = (torch.randn(rows, nh, device="cuda") * 0.5).requires_grad_(True)
    q = (
        (torch.randn(nw, nh, nq, d, device="cuda") * d**-0.5)
        .bfloat16()
        .requires_grad_(True)
    )
    k = torch.randn(nw, nh, nk, d, device="cuda").bfloat16().requires_grad_(True)
    v = torch.randn(nw, nh, nk, d, device="cuda").bfloat16().requires_grad_(True)
    grad_out = torch.randn(nw, nh, nq, d, device="cuda").bfloat16()
    leaves = (q, k, v, table)

    def grads(new: bool) -> list[Tensor]:
        for t in leaves:
            t.grad = None
        if new:
            out, lse = _flex(q, k, v, table.detach(), rpi, region)
            out = FlexBiasGrad.apply(out, table, q, k, v, lse.detach(), rpi, region)
        else:
            out = _flex(q, k, v, table * 1, rpi, region)[0]
        out.backward(grad_out)
        return [t.grad.clone() for t in leaves]  # pyright: ignore[reportOptionalMemberAccess]

    old, new = grads(False), grads(True)

    # float64 reference
    ref = [t.detach().double().requires_grad_(True) for t in leaves]
    q64, k64, v64, t64 = ref
    r = region[:, :nq].long(), region[:, :nk].long()
    mask = (r[0][:, :, None] != r[1][:, None, :]).double() * -100.0
    s = q64 @ k64.transpose(-1, -2) + t64[rpi[:nq, :nk].long()].permute(2, 0, 1)
    s = s + mask.repeat(nw // nreg, 1, 1).unsqueeze(1)
    (s.softmax(-1) @ v64).backward(grad_out.double())
    for name, a, b, r64 in zip("qkv", new[:3], old[:3], ref[:3], strict=True):
        # FlexAttention's backward without the table gradient may sum in another order
        assert rel(a, r64.grad) < 1.5 * rel(b, r64.grad) + 1e-6, name  # pyright: ignore[reportArgumentType]
    assert t64.grad is not None
    assert rel(new[3], t64.grad) < 1e-3
    assert rel(new[3], t64.grad) < 1.5 * rel(old[3], t64.grad) + 1e-6
