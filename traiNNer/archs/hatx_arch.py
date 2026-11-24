# ruff: noqa
# type: ignore
"""HAT-X: HAT-L aligned with larger windows (SST) and image-wide channel attention heads (DAT).

Experimental, for the HAT-L successor study. Built on hat_l_aligned, with three knobs:

- big_blocks / big_window: the HABs at these positions in every group (0-5) attend in
  big_window x big_window windows (shifted by half when the HAB they replace was shifted), with a
  relative position bias table of their own.
- channel: every HAB also runs DAT-style channel self-attention over the whole image on the same
  q, k and v (q and k L2-normalised along the tokens, a learned temperature per head), added to the
  window attention's output before proj through a per-channel gate that starts at 0. With Adam a
  zero-init gate barely moves at fine-tune learning rates (|g| ~1e-3 after 10k steps at 2e-5, which
  also starves q/k/temperature of gradient), so channel_gain scales it (an export folds it in).
- use_cab: False drops the HABs' conv channel-attention blocks (CAB). With cab_fade > 0 they are kept
  but their contribution fades from 1 to 0 over that many training steps (cutting them at once from a
  trained HAT-L costs ~3 dB); once it is 0 they are skipped and an export can leave them out.

Everything else (OCAB, group convs, MLPs, head, tail) is HAT-L aligned's, under the same state dict
names, so hatx_init.py builds a starting checkpoint from a HAT-L aligned one. Inputs are padded to a
multiple of the largest window.
"""

from __future__ import annotations

import torch
from spandrel.architectures.__arch_helpers.padding import pad_to_multiple
from spandrel.util.timm import trunc_normal_
from torch import nn
from torch.nn import functional as F

from traiNNer.archs.hat_arch import (
    HAB,
    HAT,
    AttnImpl,
    _flex_head_dim,
    _flex_index,
    _flex_pad,
    _flex_window,
    _flex_window_shifted,
    hat_l_aligned,
    window_partition,
    window_reverse,
)
from traiNNer.utils.registry import TESTARCH_REGISTRY


def _rpi(ws: int) -> torch.Tensor:
    c = torch.stack(
        torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
    ).flatten(1)
    r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (ws - 1)
    return r[:, :, 0] * (2 * ws - 1) + r[:, :, 1]


def _regions(h: int, w: int, ws: int, s: int) -> torch.Tensor:
    # the shifted-window region id of each token of each window (nw, ws*ws), as HAT's
    # calculate_mask_windows
    m = torch.zeros((1, h, w, 1))
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
        for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            m[:, hs, wsl, :] = cnt
            cnt += 1
    return window_partition(m, ws).view(-1, ws * ws)


class _Permute(torch.autograd.Function):
    """x[:, perm] along the tokens; the gradient is the inverse gather (no scatter-add)."""

    @staticmethod
    def forward(ctx, x, perm, inv):
        ctx.save_for_backward(inv)
        return x.index_select(1, perm)

    @staticmethod
    def backward(ctx, g):
        (inv,) = ctx.saved_tensors
        return g.index_select(1, inv), None, None


class _GramF32(torch.autograd.Function):
    """q @ k for bf16 / fp16 q, k with a float32 result (fp32 accumulation, no fp32 copies of q
    and k). The float32 gradient goes back as two half-precision GEMMs (its high and low
    halves), so it keeps float32 precision; the inner dim is only the head dim."""

    @staticmethod
    def forward(ctx, q, k):
        ctx.save_for_backward(q, k)
        return torch.bmm(q, k, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, dg):
        q, k = ctx.saved_tensors
        hi = dg.to(q.dtype)
        lo = (dg - hi.float()).to(q.dtype)
        kt, qt = k.transpose(1, 2), q.transpose(1, 2)
        dq = torch.bmm(hi, kt, out_dtype=torch.float32) + torch.bmm(
            lo, kt, out_dtype=torch.float32
        )
        dk = torch.bmm(qt, hi, out_dtype=torch.float32) + torch.bmm(
            qt, lo, out_dtype=torch.float32
        )
        return dq.to(q.dtype), dk.to(k.dtype)


class HybridAttention(nn.Module):
    """HAT's window attention (relative position bias, optional shift), plus, with channel=True, a
    DAT-style channel self-attention over the whole image on the same q, k and v, added through a
    per-channel gate that starts at 0 (so a converted HAT starts exactly where it was)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        channel: bool,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.gain = gain  # the gate is used as gate * gain: a bigger step for the zero-init gate under Adam
        self.num_heads = num_heads
        self.window_size = window_size
        self.channel = channel
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        if channel:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
            self.channel_gate = nn.Parameter(torch.zeros(dim))
        self.register_buffer("rpi", _rpi(window_size), persistent=False)
        self._cache: dict = {}

    def _region(self, h, w, s, device, flex):
        key = (h, w, s, str(device), flex)
        if key not in self._cache:
            self._cache.clear()  # one image size at a time (validation sees many)
            r = _regions(h, w, self.window_size, s)
            if flex:
                r = r.to(device=device, dtype=torch.int32)
                r = F.pad(r, (0, -r.shape[1] % 128))
            else:
                r = ((r.unsqueeze(1) - r.unsqueeze(2)).ne(0).float() * -100.0).to(
                    device
                )
            self._cache[key] = r
        return self._cache[key]

    def _window(self, qkv, h, w, shift, flex):
        ws, nh = self.window_size, self.num_heads
        d = self.dim // nh
        n = ws * ws
        if shift:
            qkv = torch.roll(qkv, shifts=(-shift, -shift), dims=(1, 2))
        t = window_partition(qkv, ws).view(-1, n, 3, nh, d).permute(2, 0, 3, 1, 4)
        q, k, v = t[0] * self.scale, t[1], t[2]  # B', nh, n, d
        if flex:
            dp = _flex_head_dim(d)
            q, k, v = (_flex_pad(z, dp) for z in (q, k, v))
            table = self.relative_position_bias_table * 1
            rpi = _flex_index(self.rpi, table.shape[0])
            if shift:
                out = _flex_window_shifted(
                    q, k, v, table, rpi, self._region(h, w, shift, q.device, True)
                )
            else:
                out = _flex_window(q, k, v, table, rpi)
            out = out[..., :d]
        else:
            bias = (
                self.relative_position_bias_table[self.rpi.view(-1)]
                .view(n, n, nh)
                .permute(2, 0, 1)
            )
            mask = self._region(h, w, shift, q.device, False) if shift else None
            outs = []
            step = max(
                1, 2**27 // (nh * n * n)
            )  # windows per chunk: scores stay ~0.5 GB in fp32
            for i in range(0, q.shape[0], step):
                j = min(i + step, q.shape[0])
                m = bias.unsqueeze(0)
                if mask is not None:
                    idx = torch.arange(i, j, device=q.device) % mask.shape[0]
                    m = m + mask[idx].unsqueeze(1)
                outs.append(
                    F.scaled_dot_product_attention(
                        q[i:j], k[i:j], v[i:j], attn_mask=m.to(q.dtype), scale=1.0
                    )
                )
            out = torch.cat(outs)
        out = out.transpose(1, 2).reshape(-1, ws, ws, nh * d)
        out = window_reverse(out, ws, h, w)
        if shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))
        return out

    def _perms(self, h, w, shift, device):
        # the image tokens in window order (rolled by -shift, then window-partitioned), and back
        key = ("perm", h, w, shift, str(device))
        if key not in self._cache:
            idx = torch.arange(h * w).view(1, h, w, 1)
            if shift:
                idx = torch.roll(idx, shifts=(-shift, -shift), dims=(1, 2))
            perm = window_partition(idx, self.window_size).reshape(-1).to(device)
            self._cache[key] = (perm, torch.argsort(perm))
        return self._cache[key]

    def _window_tokens(self, qkv, b, h, w, shift):
        """_window on FlexAttention for qkv already in window order: (b, h*w, 3c) -> (b, h*w, c)."""
        ws, nh = self.window_size, self.num_heads
        d = self.dim // nh
        t = qkv.view(-1, ws * ws, 3, nh, d).permute(2, 0, 3, 1, 4)
        q, k, v = t[0] * self.scale, t[1], t[2]
        dp = _flex_head_dim(d)
        q, k, v = (_flex_pad(z, dp) for z in (q, k, v))
        table = self.relative_position_bias_table * 1
        rpi = _flex_index(self.rpi, table.shape[0])
        if shift:
            out = _flex_window_shifted(
                q, k, v, table, rpi, self._region(h, w, shift, q.device, True)
            )
        else:
            out = _flex_window(q, k, v, table, rpi)
        return out[..., :d].transpose(1, 2).reshape(b, h * w, nh * d)

    def _channel_tokens(self, qkv, b, n):
        """_channel for (b, n, 3c) half-precision qkv -> (b, n, c): (q / |q|) . (k / |k|) as
        (q . k) / (|q| |k|), with the norms and q . k accumulated in float32 straight from qkv."""
        nh = self.num_heads
        d = self.dim // nh
        t = qkv.view(b, n, 3, nh, d)
        q = t[:, :, 0].permute(0, 2, 3, 1).reshape(b * nh, d, n)
        k = t[:, :, 1].permute(0, 2, 1, 3).reshape(b * nh, n, d)
        qn = torch.linalg.vector_norm(q, dim=-1, dtype=torch.float32).clamp_min(1e-12)
        kn = torch.linalg.vector_norm(k, dim=1, dtype=torch.float32).clamp_min(1e-12)
        g = _GramF32.apply(q, k) / (qn[:, :, None] * kn[:, None, :])
        attn = (
            (g.view(b, nh, d, d) * self.temperature).softmax(dim=-1).view(b * nh, d, d)
        )
        v = t[:, :, 2].permute(0, 2, 1, 3).reshape(b * nh, n, d)
        out = torch.bmm(v, attn.to(qkv.dtype).transpose(1, 2))  # b * nh, n, d
        return out.view(b, nh, n, d).permute(0, 2, 1, 3).reshape(b, n, self.dim)

    def _channel(self, qkv, b, h, w):
        nh = self.num_heads
        d = self.dim // nh
        t = qkv.reshape(b, h * w, 3, nh, d).permute(2, 0, 3, 4, 1)  # 3, b, nh, d, N
        q = F.normalize(t[0].float(), dim=-1)
        k = F.normalize(t[1].float(), dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(
            dim=-1
        )  # b, nh, d, d
        out = (attn.to(t.dtype) @ t[2]).permute(0, 3, 1, 2)  # b, N, nh, d
        return out.reshape(b, h, w, self.dim)

    def forward(self, x, shift, flex):
        b, h, w, c = x.shape
        if flex:
            # the qkv projection and the channel attention don't depend on the token order, so
            # the tokens go into window order once, before them, and back once after proj
            perm, inv = self._perms(h, w, shift, x.device)
            xw = _Permute.apply(x.reshape(b, h * w, c), perm, inv)
            qkv = self.qkv(xw)
            out = self._window_tokens(qkv, b, h, w, shift)
            if self.channel:
                if qkv.dtype in (torch.float16, torch.bfloat16):
                    ch = self._channel_tokens(qkv, b, h * w)
                else:
                    ch = self._channel(qkv, b, h * w, 1).reshape(b, h * w, c)
                out = out + ch * (self.channel_gate * self.gain)
            return _Permute.apply(self.proj(out), inv, perm).view(b, h, w, c)
        qkv = self.qkv(x)
        out = self._window(qkv, h, w, shift, flex)
        if self.channel:
            out = out + self._channel(qkv, b, h, w) * (self.channel_gate * self.gain)
        return self.proj(out)


class HABX(nn.Module):
    """A HAB with HybridAttention and an optional CAB; takes over the HAB's norms, MLP and CAB."""

    def __init__(
        self,
        src: HAB,
        window_size: int,
        shift_size: int,
        channel: bool,
        use_cab: bool,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = src.norm1
        self.attn = HybridAttention(src.dim, src.num_heads, window_size, channel, gain)
        self.conv_scale = src.conv_scale
        self.conv_block = src.conv_block if use_cab else None
        self.cab_mult = 1.0  # set by HATX.forward while a CAB fades out
        self.drop_path = src.drop_path
        self.norm2 = src.norm2
        self.mlp = src.mlp

    def forward(self, x, x_size, rpi_sa=None, attn_mask=None, flex=False):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        out = shortcut + self.drop_path(
            self.attn(x, self.shift_size, flex).view(b, h * w, c)
        )
        if self.conv_block is not None and self.cab_mult > 1e-3:
            conv_x = self.conv_block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            out = out + conv_x.reshape(b, h * w, c) * (self.conv_scale * self.cab_mult)
        return out + self.drop_path(self.mlp(self.norm2(out)))


class HATX(HAT):
    pad_multiple = 16
    cab_fade = 0

    def check_image_size(self, x):
        return pad_to_multiple(x, self.pad_multiple, mode="reflect")

    def forward(self, x):
        if self.cab_fade:
            # the CABs' contribution goes from 1 to 0 over cab_fade training steps; a float buffer, so
            # checkpoints resume it and the EMA copy follows it (lerped, a little behind)
            if self.training:
                self.cab_mult.sub_(1.0 / self.cab_fade).clamp_(min=0.0)
            mult = float(self.cab_mult)
            for m in self.modules():
                if isinstance(m, HABX):
                    m.cab_mult = mult
        return super().forward(x)


@TESTARCH_REGISTRY.register()
def hatx_l(
    scale: int = 4,
    big_blocks: tuple[int, ...] = (),
    big_window: int = 32,
    channel: bool = False,
    channel_gain: float = 1.0,
    use_cab: bool = True,
    cab_fade: int = 0,
    drop_path_rate: float = 0.1,
    use_checkpoint: bool = False,
    attn_impl: AttnImpl = "auto",
) -> HATX:
    net = hat_l_aligned(
        scale=scale,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        attn_impl=attn_impl,
    )
    net.__class__ = HATX
    for layer in net.layers:
        blocks = layer.residual_group.blocks
        for j, blk in enumerate(blocks):
            big = j in big_blocks
            if not (big or channel or not use_cab):
                continue
            ws = big_window if big else blk.window_size
            keep_cab = use_cab or cab_fade > 0
            blocks[j] = HABX(
                blk,
                ws,
                ws // 2 if blk.shift_size > 0 else 0,
                channel,
                keep_cab,
                channel_gain,
            )
    net.pad_multiple = max(net.window_size, big_window if big_blocks else 0)
    if not use_cab and cab_fade > 0:
        net.cab_fade = cab_fade
        net.register_buffer("cab_mult", torch.ones(()))
    return net
