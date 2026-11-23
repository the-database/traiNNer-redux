"""Gradient averaging across GPUs inside the training step, for training on several GPUs
with ``use_cuda_graph``.

DistributedDataParallel averages gradients from autograd hooks while the backward runs,
from Python, and can't follow a step that turns the discriminator's ``requires_grad``
off and on. GradSync does what it does as plain collectives inside the step, so the
whole step, collectives included, can be recorded as one CUDA graph:

- At the start it broadcasts every parameter and buffer from rank 0, as DDP does when it
  wraps a model (each rank seeds its own initialisation).
- Every gradient is a view into one flat buffer per network and dtype (DDP's
  ``gradient_as_bucket_view``), so one all-reduce averages them all. It runs after the
  backward of the micro-batch that applies the update, before unscaling, clipping and the
  optimizer step. With gradient accumulation the accumulated gradients are averaged once,
  which is the same average as DDP's per-micro-batch one.
- At the start of each step it broadcasts rank 0's buffers, as DDP does, but only the
  buffers that change while training (BatchNorm statistics, spectral norm vectors,
  ...). The others are constants, the same on every rank, and can be large (DAT keeps
  72 MB of attention masks). The buffers that changed over the first step are the ones
  kept, agreed across the ranks.

Unlike DDP, the all-reduce doesn't overlap the backward.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist
from torch import Tensor, nn


def _dense(t: Tensor) -> bool:
    return t.is_contiguous() or t.is_contiguous(memory_format=torch.channels_last)


class _NetSync:
    def __init__(self, net: nn.Module) -> None:
        self.net = net
        # gradients: one flat buffer per dtype, each parameter's .grad a view into it
        self.params: list[tuple[nn.Parameter, Tensor]] = []
        self.loose: list[nn.Parameter] = []  # neither contiguous nor channels_last
        by_dtype: dict[torch.dtype, list[nn.Parameter]] = defaultdict(list)
        for p in net.parameters():
            if p.requires_grad:
                if _dense(p):
                    by_dtype[p.dtype].append(p)
                else:
                    self.loose.append(p)
        self.flat_grads: list[Tensor] = []
        for dtype, params in by_dtype.items():
            flat = torch.zeros(
                sum(p.numel() for p in params), dtype=dtype, device=params[0].device
            )
            offset = 0
            for p in params:
                view = flat.as_strided(p.shape, p.stride(), offset)
                if p.grad is not None:
                    view.copy_(p.grad)
                p.grad = view
                self.params.append((p, view))
                offset += p.numel()
            self.flat_grads.append(flat)
        for p in self.loose:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        # buffers: the float ones, as they were before training changed any
        self.candidates = [
            (m, name)
            for m in net.modules()
            for name, b in m._buffers.items()  # noqa: SLF001
            if b is not None and b.is_floating_point()
        ]
        self.snapshot: list[Tensor] | None = [
            getattr(m, name).clone() for m, name in self.candidates
        ]
        self.steps = 0
        # per dtype: the (module, name) of the buffers kept, a flat buffer, its views
        self.buffers: list[
            tuple[list[tuple[nn.Module, str]], Tensor, list[Tensor]]
        ] = []


class GradSync:
    """Averages gradients and broadcasts buffers across the ranks inside the training
    step (see the module docstring). Use one for all networks trained in the step."""

    def __init__(self, nets: list[nn.Module]) -> None:
        self.world_size = dist.get_world_size()
        self._nets: dict[int, _NetSync] = {}
        for net in nets:
            with torch.no_grad():
                for t in list(net.parameters()) + list(net.buffers()):
                    dist.broadcast(t.data, 0)
            self._nets[id(net)] = _NetSync(net)

    def all_reduce_grads(self, net: nn.Module) -> None:
        """Average ``net``'s gradients across the ranks, in place."""
        s = self._nets[id(net)]
        for p, view in s.params:
            if p.grad is not view:
                # something replaced the gradient (e.g. zero_grad(set_to_none=True)):
                # move it back into the flat buffer
                if p.grad is None:
                    view.zero_()
                else:
                    view.copy_(p.grad)
                p.grad = view
        for flat in s.flat_grads:
            dist.all_reduce(flat)
            flat.div_(self.world_size)
        for p in s.loose:
            assert p.grad is not None
            dist.all_reduce(p.grad)
            p.grad.div_(self.world_size)

    def sync_buffers(self, net: nn.Module) -> None:
        """Give every rank rank 0's buffers that change while training. Call it at the
        start of each step; the first two calls (before the second step) run eagerly."""
        s = self._nets[id(net)]
        if s.snapshot is not None:
            s.steps += 1
            if s.steps < 2:
                return
            self._select_buffers(s)
        for names, flat, views in s.buffers:
            bufs = [getattr(m, name) for m, name in names]
            torch._foreach_copy_(views, bufs)  # pyright: ignore[reportPrivateImportUsage]
            dist.broadcast(flat, 0)
            torch._foreach_copy_(bufs, views)  # pyright: ignore[reportPrivateImportUsage]

    def _select_buffers(self, s: _NetSync) -> None:
        # the buffers the first step changed on any rank: every rank must broadcast the
        # same set, or the collectives wouldn't match
        assert s.snapshot is not None
        device = next(s.net.parameters()).device
        changed = torch.tensor(
            [
                not torch.equal(getattr(m, name), snap)
                for (m, name), snap in zip(s.candidates, s.snapshot, strict=True)
            ],
            dtype=torch.int32,
            device=device,
        )
        if changed.numel():
            dist.all_reduce(changed, op=dist.ReduceOp.MAX)
        keep = [
            c for c, flag in zip(s.candidates, changed.tolist(), strict=True) if flag
        ]
        s.snapshot = None
        by_dtype: dict[torch.dtype, list[tuple[nn.Module, str]]] = defaultdict(list)
        for m, name in keep:
            by_dtype[getattr(m, name).dtype].append((m, name))
        for dtype, names in by_dtype.items():
            bufs = [getattr(m, name) for m, name in names]
            flat = torch.empty(sum(b.numel() for b in bufs), dtype=dtype, device=device)
            views = [
                v.view(b.shape)
                for v, b in zip(
                    flat.split([b.numel() for b in bufs]), bufs, strict=True
                )
            ]
            s.buffers.append((names, flat, views))
