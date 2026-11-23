# ruff: noqa: ANN001, ANN202, N803  (Triton kernels: untyped pointers, uppercase arguments)
# pyright: reportCallIssue=false
"""The relative position bias gradient of a FlexAttention window attention, without atomics.

With ``score_mod = score + table[rpi[i, j], h]``, FlexAttention's backward adds every score's
gradient into the bias table with an atomic add: windows x heads x n x n atomics into a table of a
few thousand entries, which took more than half of HAT's attention backward. Here FlexAttention
computes q, k and v's gradients with the table detached, and a kernel sums the score gradients,
``dS = P * (dO.V^T - rowsum(dO * O))`` with ``P = exp(S - lse)``, over the windows in registers: one
program per head and score tile, looping over the windows (split into a few chunks when there are
few tiles). The [heads, n, n] sum is then gathered into the table. The same gradient, with the
same accuracy; q, k and v's gradients are unchanged.
"""

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _bias_grad_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    table_ptr,
    rpi_ptr,
    region_ptr,
    out_ptr,
    NW,
    NQ,
    NK,
    NREG,
    B_PER,
    sqb,
    sqh,
    sqn,
    skb,
    skh,
    skn,
    svb,
    svh,
    svn,
    sdb,
    sdh,
    sdn,
    slb,
    slh,
    stab,
    srpi,
    sreg,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_REGION: tl.constexpr,
    PRECISION: tl.constexpr,
):
    tile = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)
    n_tn = tl.cdiv(NK, BLOCK_N)
    offs_i = (tile // n_tn) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = (tile % n_tn) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mi = offs_i < NQ
    mj = offs_j < NK
    mij = mi[:, None] & mj[None, :]
    idx = tl.load(rpi_ptr + offs_i[:, None] * srpi + offs_j[None, :], mask=mij, other=0)
    bias = tl.load(table_ptr + idx * stab + h, mask=mij, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b_start = split * B_PER
    b_end = tl.minimum(b_start + B_PER, NW)
    for b in range(b_start, b_end):
        q = tl.load(
            q_ptr + b * sqb + h * sqh + offs_i[:, None] * sqn + offs_d[None, :],
            mask=mi[:, None],
            other=0.0,
        )
        k = tl.load(
            k_ptr + b * skb + h * skh + offs_j[:, None] * skn + offs_d[None, :],
            mask=mj[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) + bias
        if (
            HAS_REGION
        ):  # the shifted-window mask: -100 between tokens of different regions
            w = b % NREG
            ri = tl.load(region_ptr + w * sreg + offs_i, mask=mi, other=0)
            rj = tl.load(region_ptr + w * sreg + offs_j, mask=mj, other=0)
            s = s + tl.where(ri[:, None] == rj[None, :], 0.0, -100.0)
        lse = tl.load(lse_ptr + b * slb + h * slh + offs_i, mask=mi, other=0.0)
        p = tl.math.exp2((s - lse[:, None]) * 1.4426950408889634)  # exp(s - lse)
        do = tl.load(
            do_ptr + b * sdb + h * sdh + offs_i[:, None] * sdn + offs_d[None, :],
            mask=mi[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + b * svb + h * svh + offs_j[:, None] * svn + offs_d[None, :],
            mask=mj[:, None],
            other=0.0,
        )
        dp = tl.dot(do, tl.trans(v), input_precision=PRECISION)
        delta = tl.load(delta_ptr + b * slb + h * slh + offs_i, mask=mi, other=0.0)
        acc += p * (dp - delta[:, None])
    out = out_ptr + (split * tl.num_programs(1) + h) * NQ * NK
    tl.store(out + offs_i[:, None] * NK + offs_j[None, :], acc, mask=mij)


def _last_contiguous(t: Tensor) -> Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def flex_bias_grad(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    grad_out: Tensor,
    lse: Tensor,
    table: Tensor,
    rpi: Tensor,
    region: Tensor | None = None,
) -> Tensor:
    """d loss / d table for ``flex_attention(q, k, v, scale=1.0)`` with ``score_mod = score +
    table[rpi[i, j], h]``, plus ``-100`` where ``region[b % len(region), i] != region[..., j]``
    when region is given. q: [windows, heads, nq, d], k and v: [windows, heads, nk, d], d a power
    of two of at least 16; out, grad_out: the attention output and its gradient; lse: the
    natural-log logsumexp FlexAttention returned, [windows, heads, nq]; table: [rows, heads];
    rpi: [>= nq, >= nk] indices into table, in range; region: [region windows, >= n] ints."""
    nw, nh, nq, d = q.shape
    nk = k.shape[2]
    q, k, v, grad_out = (_last_contiguous(t) for t in (q, k, v, grad_out))
    delta = (grad_out.float() * out.float()).sum(-1)
    lse = lse.float().contiguous()
    rpi = rpi.to(torch.int32)
    reg = rpi if region is None else region.to(torch.int32)
    block = 64
    tiles = triton.cdiv(nq, block) * triton.cdiv(nk, block)
    # enough programs to fill the GPU: split the windows into chunks summed afterwards
    n_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
    b_per = triton.cdiv(nw, max(1, min(nw, triton.cdiv(4 * n_sm, tiles * nh))))
    split = triton.cdiv(nw, b_per)
    part = torch.empty(split, nh, nq, nk, device=q.device, dtype=torch.float32)
    _bias_grad_kernel[(tiles, nh, split)](
        q,
        k,
        v,
        grad_out,
        lse,
        delta,
        table,
        rpi,
        reg,
        part,
        nw,
        nq,
        nk,
        reg.shape[0],
        b_per,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        lse.stride(0),
        lse.stride(1),
        table.stride(0),
        rpi.stride(0),
        reg.stride(0),
        D=d,
        BLOCK_M=block,
        BLOCK_N=block,
        HAS_REGION=region is not None,
        PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
        num_warps=4,
        num_stages=2,
    )
    dense = part.sum(0) if split > 1 else part[0]  # heads, nq, nk
    grad = torch.zeros(table.shape[0], nh, device=q.device, dtype=torch.float32)
    grad.index_add_(
        0, rpi[:nq, :nk].reshape(-1), dense.permute(1, 2, 0).reshape(-1, nh)
    )
    return grad.to(table.dtype)


class FlexBiasGrad(torch.autograd.Function):
    """The identity on a FlexAttention output whose backward gives the bias table's gradient:
    ``out = FlexBiasGrad.apply(flex_out, table, q, k, v, lse, rpi, region)``, with flex_out from a
    flex_attention call that used ``table.detach()``."""

    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        out: Tensor,
        table: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        lse: Tensor,
        rpi: Tensor,
        region: Tensor | None,
    ) -> Tensor:
        ctx.save_for_backward(out, table, q, k, v, lse, rpi, region)
        return out.view_as(out)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad_out: Tensor
    ) -> tuple[Tensor | None, ...]:
        out, table, q, k, v, lse, rpi, region = ctx.saved_tensors
        grad_table = flex_bias_grad(q, k, v, out, grad_out, lse, table, rpi, region)
        return grad_out, grad_table, None, None, None, None, None, None
