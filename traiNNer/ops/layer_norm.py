# ruff: noqa: ANN001, ANN202, N803  (Triton kernels: untyped pointers, uppercase constexpr)
# pyright: reportCallIssue=false, reportPossiblyUnboundVariable=false
"""LayerNorm over the last dimension as Triton kernels.

PyTorch's LayerNorm backward is slow for the shapes transformer SR archs train
with, such as DAT's 32,768 tokens of 180 channels: its weight and bias gradient
reduction alone took 26 ms of a DAT_2 step. These kernels handle tiles of rows per
program and reduce the weight and bias gradients in two stages.

Semantics match ``F.layer_norm``: statistics and arithmetic in float32, and under
autocast the output is float32 (autocast runs layer_norm in float32). The input is
read in its own dtype instead of being cast to float32 first.
"""

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit
def _ln_fwd(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    n_rows,
    n_cols,
    eps,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None].to(tl.int64) * n_cols + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / n_cols
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd[:, None]
    if HAS_W:
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = y * w[None, :]
    if HAS_B:
        b = tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = y + b[None, :]
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)


@triton.jit
def _ln_bwd(
    dy_ptr,
    x_ptr,
    w_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dw_part_ptr,
    db_part_ptr,
    n_rows,
    n_cols,
    HAS_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROW_TILES: tl.constexpr,
):
    # each program handles ROW_TILES tiles of BLOCK_M rows and writes one partial
    # sum of the weight and bias gradients
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols
    if HAS_W:
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for t in range(ROW_TILES):
        rows = (pid * ROW_TILES + t) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None].to(tl.int64) * n_cols + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        dw_acc += tl.sum(dy * xhat, axis=0)
        db_acc += tl.sum(dy, axis=0)
        wdy = dy * w[None, :] if HAS_W else dy
        c1 = tl.sum(xhat * wdy, axis=1) / n_cols
        c2 = tl.sum(wdy, axis=1) / n_cols
        dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dw_part_ptr + pid * n_cols + cols, dw_acc, mask=col_mask)
    tl.store(db_part_ptr + pid * n_cols + cols, db_acc, mask=col_mask)


# Tile sizes, tuned on DAT's 32768 x 180 (BLOCK_N 256) on an RTX 5090: the forward
# runs 16 x 256 tiles with 2 warps, the backward 8 x 256 tiles with 4 warps, four
# tiles per program. Other widths keep the same number of elements per tile.
def _fwd_block_m(block_n: int) -> int:
    return max(1, min(128, 4096 // block_n))


def _bwd_block_m(block_n: int) -> int:
    return max(1, min(64, 2048 // block_n))


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        x: Tensor,
        weight: Tensor | None,
        bias: Tensor | None,
        eps: float,
        out_dtype: torch.dtype,
    ) -> Tensor:
        n_cols = x.shape[-1]
        x2 = x.reshape(-1, n_cols)
        n_rows = x2.shape[0]
        # allocated in x's shape and returned as is: a view output can't be
        # modified in place, which some archs do
        y = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        block_n = triton.next_power_of_2(n_cols)
        block_m = _fwd_block_m(block_n)
        grid = (triton.cdiv(n_rows, block_m),)
        _ln_fwd[grid](
            x2,
            weight if weight is not None else x2,
            bias if bias is not None else x2,
            y.view(n_rows, n_cols),
            mean,
            rstd,
            n_rows,
            n_cols,
            eps,
            HAS_W=weight is not None,
            HAS_B=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=2 if block_n <= 256 else (4 if block_n <= 1024 else 8),
        )
        ctx.save_for_backward(x2, weight, mean, rstd)  # pyright: ignore[reportArgumentType]
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, dy: Tensor
    ) -> tuple[Tensor | None, ...]:
        x2, weight, mean, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        dy2 = dy.reshape(-1, n_cols).contiguous()
        dx = torch.empty(ctx.x_shape, dtype=ctx.x_dtype, device=x2.device)
        block_n = triton.next_power_of_2(n_cols)
        block_m = _bwd_block_m(block_n)
        row_tiles = 4
        n_prog = triton.cdiv(n_rows, block_m * row_tiles)
        dw_part = torch.empty((n_prog, n_cols), dtype=torch.float32, device=x2.device)
        db_part = torch.empty((n_prog, n_cols), dtype=torch.float32, device=x2.device)
        _ln_bwd[(n_prog,)](
            dy2,
            x2,
            weight if weight is not None else x2,
            mean,
            rstd,
            dx.view(n_rows, n_cols),
            dw_part,
            db_part,
            n_rows,
            n_cols,
            HAS_W=weight is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            ROW_TILES=row_tiles,
            num_warps=4 if block_n <= 1024 else 8,
        )
        dw = dw_part.sum(0).to(weight.dtype) if weight is not None else None
        db = (
            db_part.sum(0).to(weight.dtype if weight is not None else torch.float32)
            if ctx.has_bias
            else None
        )
        return dx, dw, db, None, None


def layer_norm(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float = 1e-5,
) -> Tensor:
    """``F.layer_norm(x, (x.shape[-1],), weight, bias, eps)`` with Triton kernels."""
    out_dtype = (
        torch.float32
        if torch.is_autocast_enabled(x.device.type)
        and x.dtype in (torch.float16, torch.bfloat16)
        else x.dtype
    )
    if not x.is_contiguous():
        x = x.contiguous()
    with torch.autocast(x.device.type, enabled=False):
        return _LayerNorm.apply(x, weight, bias, eps, out_dtype)  # pyright: ignore[reportReturnType]


# Wider rows don't fit the one-tile-per-row kernels; they keep PyTorch's version.
_MAX_COLS = 8192


class TritonLayerNorm(nn.LayerNorm):
    """nn.LayerNorm whose forward uses the Triton kernels where they apply.

    Same parameters and state dict as nn.LayerNorm. The kernels are much faster on
    the GPU but slower to launch from Python, so this pays off when the launches
    are recorded once, in a CUDA graph (use_cuda_graph), and not in eager training.
    """

    def forward(self, input: Tensor) -> Tensor:
        x = input
        if (
            x.is_cuda
            and len(self.normalized_shape) == 1
            and x.shape[-1] == self.normalized_shape[0]
            and x.shape[-1] <= _MAX_COLS
        ):
            return layer_norm(x, self.weight, self.bias, self.eps)
        return super().forward(x)


def use_triton_layer_norm(net: nn.Module) -> int:
    """Switch every nn.LayerNorm in ``net`` to TritonLayerNorm in place, and return
    how many were switched. Subclasses of nn.LayerNorm are left alone."""
    n = 0
    for m in net.modules():
        if type(m) is nn.LayerNorm:
            m.__class__ = TritonLayerNorm
            n += 1
    return n
