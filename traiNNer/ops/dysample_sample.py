# ruff: noqa: ANN001, ANN202, N803, N806  (Triton kernels: untyped pointers, uppercase constexpr)
# pyright: reportCallIssue=false, reportPossiblyUnboundVariable=false
"""DySample's sampling step, ``F.grid_sample(mode="bilinear", padding_mode="border",
align_corners=False)`` on channel groups, as Triton kernels for channels_last input.

PyTorch runs grid_sample in float32 under autocast on a contiguous tensor, so
DySample had to split the channel groups out of a channels_last input with a copy,
cast it to float32 with another, and convert the output back for the next
convolution, and the same again in backward. These kernels read the input in its own
dtype and layout, compute in float32, and write the output in the dtype and layout
asked for. The backward adds input gradients with float32 atomics over contiguous
channels and sums the sampling-grid gradient inside each program.
"""

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _source_index(coord, size):
    # unnormalize for align_corners=False, then clip for border padding; returns the
    # index and d(index)/d(coord), which is 0 where the coordinate was clipped
    ix = ((coord + 1.0) * size - 1.0) * 0.5
    mult = tl.full(ix.shape, 0.5, tl.float32) * size
    low = ix <= 0.0
    high = ix >= size - 1.0
    ix = tl.where(low, 0.0, tl.where(high, size - 1.0, ix))
    mult = tl.where(low | high, 0.0, mult)
    return ix, mult


@triton.jit
def _dysample_fwd(
    x_ptr,
    grid_ptr,
    out_ptr,
    H,
    W,
    OH,
    OW,
    CG,
    sxb,
    sxc,
    sxh,
    sxw,
    sob,
    soc,
    soh,
    sow,
    n_pix,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    G: tl.constexpr,
):
    pid_p = tl.program_id(0)
    g = tl.program_id(1)
    pix = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    pmask = pix < n_pix
    b = pix // (OH * OW)
    rem = pix % (OH * OW)
    oy = rem // OW
    ox = rem % OW
    gbase = ((b * G + g) * OH + oy) * OW + ox
    gx = tl.load(grid_ptr + gbase * 2, mask=pmask, other=0.0)
    gy = tl.load(grid_ptr + gbase * 2 + 1, mask=pmask, other=0.0)
    ix, _ = _source_index(gx, W * 1.0)
    iy, _ = _source_index(gy, H * 1.0)
    ix0f = tl.floor(ix)
    iy0f = tl.floor(iy)
    ix0 = ix0f.to(tl.int32)
    iy0 = iy0f.to(tl.int32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    w_nw = (ix0f + 1.0 - ix) * (iy0f + 1.0 - iy)
    w_ne = (ix - ix0f) * (iy0f + 1.0 - iy)
    w_sw = (ix0f + 1.0 - ix) * (iy - iy0f)
    w_se = (ix - ix0f) * (iy - iy0f)
    in_x1 = ix1 < W
    in_y1 = iy1 < H
    xb = b.to(tl.int64) * sxb
    ob = b.to(tl.int64) * sob + oy.to(tl.int64) * soh + ox.to(tl.int64) * sow
    for c0 in range(0, CG, BLOCK_C):
        cs = c0 + tl.arange(0, BLOCK_C)
        cmask = cs < CG
        ch = (g * CG + cs).to(tl.int64) * sxc
        m = pmask[:, None] & cmask[None, :]
        base_nw = xb + iy0.to(tl.int64) * sxh + ix0.to(tl.int64) * sxw
        v_nw = tl.load(x_ptr + base_nw[:, None] + ch[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        v_ne = tl.load(
            x_ptr + (base_nw + sxw)[:, None] + ch[None, :],
            mask=m & in_x1[:, None],
            other=0.0,
        ).to(tl.float32)
        v_sw = tl.load(
            x_ptr + (base_nw + sxh)[:, None] + ch[None, :],
            mask=m & in_y1[:, None],
            other=0.0,
        ).to(tl.float32)
        v_se = tl.load(
            x_ptr + (base_nw + sxh + sxw)[:, None] + ch[None, :],
            mask=m & (in_x1 & in_y1)[:, None],
            other=0.0,
        ).to(tl.float32)
        # accumulated in the order PyTorch's grid_sample kernel uses, so the result
        # is bit-identical to it
        val = v_nw * w_nw[:, None]
        val = tl.fma(v_ne, w_ne[:, None], val)
        val = tl.fma(v_sw, w_sw[:, None], val)
        val = tl.fma(v_se, w_se[:, None], val)
        och = (g * CG + cs).to(tl.int64) * soc
        tl.store(
            out_ptr + ob[:, None] + och[None, :],
            val.to(out_ptr.dtype.element_ty),
            mask=m,
        )


@triton.jit
def _dysample_bwd(
    gout_ptr,
    x_ptr,
    grid_ptr,
    gx_ptr,
    ggrid_ptr,
    H,
    W,
    OH,
    OW,
    CG,
    sxb,
    sxc,
    sxh,
    sxw,
    sgb,
    sgc,
    sgh,
    sgw,
    n_pix,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    G: tl.constexpr,
):
    pid_p = tl.program_id(0)
    g = tl.program_id(1)
    pix = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    pmask = pix < n_pix
    b = pix // (OH * OW)
    rem = pix % (OH * OW)
    oy = rem // OW
    ox = rem % OW
    gbase = ((b * G + g) * OH + oy) * OW + ox
    gxc = tl.load(grid_ptr + gbase * 2, mask=pmask, other=0.0)
    gyc = tl.load(grid_ptr + gbase * 2 + 1, mask=pmask, other=0.0)
    ix, mult_x = _source_index(gxc, W * 1.0)
    iy, mult_y = _source_index(gyc, H * 1.0)
    ix0f = tl.floor(ix)
    iy0f = tl.floor(iy)
    ix0 = ix0f.to(tl.int32)
    iy0 = iy0f.to(tl.int32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    w_nw = (ix0f + 1.0 - ix) * (iy0f + 1.0 - iy)
    w_ne = (ix - ix0f) * (iy0f + 1.0 - iy)
    w_sw = (ix0f + 1.0 - ix) * (iy - iy0f)
    w_se = (ix - ix0f) * (iy - iy0f)
    in_x1 = ix1 < W
    in_y1 = iy1 < H
    xb = b.to(tl.int64) * sxb
    # gx is a contiguous float32 (B, H, W, C) buffer
    C = CG * G
    gxb = b.to(tl.int64) * H * W * C
    gob = b.to(tl.int64) * sgb + oy.to(tl.int64) * sgh + ox.to(tl.int64) * sgw
    gix = tl.zeros((BLOCK_P,), dtype=tl.float32)
    giy = tl.zeros((BLOCK_P,), dtype=tl.float32)
    base_nw = xb + iy0.to(tl.int64) * sxh + ix0.to(tl.int64) * sxw
    gbase_nw = gxb + (iy0.to(tl.int64) * W + ix0.to(tl.int64)) * C
    for c0 in range(0, CG, BLOCK_C):
        cs = c0 + tl.arange(0, BLOCK_C)
        cmask = cs < CG
        m = pmask[:, None] & cmask[None, :]
        chx = (g * CG + cs).to(tl.int64) * sxc
        chg = (g * CG + cs).to(tl.int64)
        go = tl.load(
            gout_ptr + gob[:, None] + (chg * sgc)[None, :], mask=m, other=0.0
        ).to(tl.float32)
        m_ne = m & in_x1[:, None]
        m_sw = m & in_y1[:, None]
        m_se = m & (in_x1 & in_y1)[:, None]
        v_nw = tl.load(x_ptr + base_nw[:, None] + chx[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        v_ne = tl.load(
            x_ptr + (base_nw + sxw)[:, None] + chx[None, :], mask=m_ne, other=0.0
        ).to(tl.float32)
        v_sw = tl.load(
            x_ptr + (base_nw + sxh)[:, None] + chx[None, :], mask=m_sw, other=0.0
        ).to(tl.float32)
        v_se = tl.load(
            x_ptr + (base_nw + sxh + sxw)[:, None] + chx[None, :], mask=m_se, other=0.0
        ).to(tl.float32)
        tl.atomic_add(
            gx_ptr + gbase_nw[:, None] + chg[None, :],
            w_nw[:, None] * go,
            mask=m,
            sem="relaxed",
        )
        tl.atomic_add(
            gx_ptr + (gbase_nw + C)[:, None] + chg[None, :],
            w_ne[:, None] * go,
            mask=m_ne,
            sem="relaxed",
        )
        tl.atomic_add(
            gx_ptr + (gbase_nw + W * C)[:, None] + chg[None, :],
            w_sw[:, None] * go,
            mask=m_sw,
            sem="relaxed",
        )
        tl.atomic_add(
            gx_ptr + (gbase_nw + W * C + C)[:, None] + chg[None, :],
            w_se[:, None] * go,
            mask=m_se,
            sem="relaxed",
        )
        dy0 = iy0f + 1.0 - iy
        dy1 = iy - iy0f
        dx0 = ix0f + 1.0 - ix
        dx1 = ix - ix0f
        gix += tl.sum(
            go
            * (
                -v_nw * dy0[:, None]
                + v_ne * dy0[:, None]
                - v_sw * dy1[:, None]
                + v_se * dy1[:, None]
            ),
            axis=1,
        )
        giy += tl.sum(
            go
            * (
                -v_nw * dx0[:, None]
                - v_ne * dx1[:, None]
                + v_sw * dx0[:, None]
                + v_se * dx1[:, None]
            ),
            axis=1,
        )
    tl.store(ggrid_ptr + gbase * 2, gix * mult_x, mask=pmask)
    tl.store(ggrid_ptr + gbase * 2 + 1, giy * mult_y, mask=pmask)


def _blocks(cg: int) -> tuple[int, int]:
    block_c = min(64, triton.next_power_of_2(cg))
    block_p = max(16, 4096 // block_c // 4)
    return block_p, block_c


class _DySampleSample(torch.autograd.Function):
    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        x: Tensor,
        grid: Tensor,
        groups: int,
        out_dtype: torch.dtype,
        out_format: torch.memory_format,
    ) -> Tensor:
        B, C, H, W = x.shape
        _, OH, OW, _ = grid.shape
        cg = C // groups
        out = torch.empty(
            (B, C, OH, OW), dtype=out_dtype, device=x.device, memory_format=out_format
        )
        n_pix = B * OH * OW
        block_p, block_c = _blocks(cg)
        _dysample_fwd[(triton.cdiv(n_pix, block_p), groups)](
            x, grid, out, H, W, OH, OW, cg,
            *x.stride(), *out.stride(),
            n_pix, BLOCK_P=block_p, BLOCK_C=block_c, G=groups, num_warps=4,
        )  # fmt: skip
        ctx.save_for_backward(x, grid)
        ctx.groups = groups
        return out

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, gout: Tensor
    ) -> tuple[Tensor | None, ...]:
        x, grid = ctx.saved_tensors
        groups = ctx.groups
        B, C, H, W = x.shape
        _, OH, OW, _ = grid.shape
        cg = C // groups
        # float32 accumulation buffer, (B, H, W, C) in memory
        gx = torch.zeros((B, H, W, C), dtype=torch.float32, device=x.device)
        ggrid = torch.empty_like(grid)
        n_pix = B * OH * OW
        block_p, block_c = _blocks(cg)
        _dysample_bwd[(triton.cdiv(n_pix, block_p), groups)](
            gout, x, grid, gx, ggrid, H, W, OH, OW, cg,
            *x.stride(), *gout.stride(),
            n_pix, BLOCK_P=block_p, BLOCK_C=block_c, G=groups, num_warps=4,
        )  # fmt: skip
        gx = gx.permute(0, 3, 1, 2).to(
            dtype=x.dtype, memory_format=torch.preserve_format
        )
        return gx, ggrid, None, None, None


def dysample_sample(
    x: Tensor,
    grid: Tensor,
    groups: int,
    out_dtype: torch.dtype,
    out_format: torch.memory_format,
) -> Tensor:
    """Bilinear border-padded grid_sample of x's channel groups.

    x: (B, C, H, W) in any layout and float dtype. grid: (B * groups, OH, OW, 2)
    float32 in [-1, 1], as F.grid_sample takes it for x split into groups along
    channels. Returns (B, C, OH, OW) in ``out_dtype`` and ``out_format``.
    """
    with torch.autocast(x.device.type, enabled=False):
        return _DySampleSample.apply(
            x, grid.contiguous(), groups, out_dtype, out_format
        )  # pyright: ignore[reportReturnType]
