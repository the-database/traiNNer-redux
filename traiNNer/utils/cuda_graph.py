"""Record the whole training step as a CUDA graph and replay it.

Launching the kernels of a training step one by one from Python costs more than
running them for many archs: eager DAT spends about 780 ms per step on about 230 ms
of GPU work. A CUDA graph records the forward, backward and optimizer step once and
then launches them all with one call. Unlike torch.compile it takes seconds and
leaves no cache on disk.

A step can be recorded when it has fixed input shapes and does no CPU work that
depends on the GPU (no ``.item()``, no CPU-to-GPU copies). Anything else is caught
when recording, and training carries on without graphs.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler, _refresh_per_optimizer_state
from torch.optim import Optimizer

from traiNNer.utils.logger import get_root_logger

T = TypeVar("T")

# group settings the recorded optimizer step bakes in, besides lr
_FIXED_HPARAMS = ("betas", "momentum", "weight_decay", "eps")


def supports_cuda_graph(optimizer: Optimizer) -> str | None:
    """Why ``optimizer`` can't be recorded, or None when it can.

    The step must skip a non-finite update on the GPU, which GradScaler only does
    for optimizers with a fused step, and the optimizer must be capturable.
    """
    name = optimizer.__class__.__name__
    if not getattr(optimizer, "_step_supports_amp_scaling", False):
        return f"{name} has no fused step"
    if not all(g.get("capturable") for g in optimizer.param_groups):
        return f"{name} is not capturable"
    return None


def move_tensor_attributes(net: nn.Module, device: torch.device) -> None:
    """Move plain tensor attributes (not parameters or buffers) of ``net`` to
    ``device``.

    Several archs keep a constant such as the RGB mean as a plain CPU tensor and copy
    it to the GPU on every forward. The copy can't be recorded in a CUDA graph. These
    attributes aren't in the state dict, so moving them changes no saved weights.
    """
    for module in net.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, Tensor) and value.device != device:
                setattr(module, name, value.to(device))


def _clear_capture_error() -> None:
    """A failed recording leaves its CUDA error pending, and the next kernel launch
    reports it once. Launch a throwaway kernel to take it."""
    for _ in range(3):
        try:
            torch.zeros((), device="cuda").add_(1)
            torch.cuda.synchronize()
            return
        except Exception:  # noqa: BLE001, S112
            continue


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    inputs: dict[str, Tensor]
    outputs: Any
    hparams: list[tuple[Any, ...]]


class CudaGraphStep:
    """Runs a training step function eagerly for a few warmup calls, then records it
    as a CUDA graph and replays the graph on every later call with the same key.

    The step function takes a dict of input tensors and returns its outputs. When
    replaying, the inputs are copied into the recorded input tensors and the outputs
    are the same tensors on every call, overwritten by each replay. Use them before
    the next call.

    Requirements on the step function:

    - It must leave gradients allocated (``zero_grad(set_to_none=False)``), so the
      recorded backward accumulates into the same gradient tensors on every replay.
    - State it keeps between steps must be updated in place, since attribute
      assignments only happen while recording.
    - Python-side decisions inside it must depend only on the key.

    The optimizers' learning rates stay Python floats, so schedulers and checkpoints
    work as usual: each replay copies the current lr into the tensor the recorded
    step reads.
    """

    def __init__(
        self,
        optimizers: list[Optimizer],
        scalers: list[GradScaler],
        warmup_steps: int = 3,
        max_graphs: int = 4,
    ) -> None:
        self.optimizers = optimizers
        self.scalers = scalers
        self.warmup_steps = warmup_steps
        self.max_graphs = max_graphs
        self.enabled = True
        self._graphs: dict[Hashable, _Graph] = {}
        self._inputs: dict[Hashable, dict[str, Tensor]] = {}
        self._calls: dict[Hashable, int] = defaultdict(int)
        self._pool: Any = None
        self._side_stream = torch.cuda.Stream()
        self._lr_tensors = [
            [
                torch.tensor(float(g["lr"]), dtype=torch.float32, device="cuda")
                for g in o.param_groups
            ]
            for o in optimizers
        ]
        self._warned_shapes = False

    def __call__(
        self,
        key: Hashable,
        inputs: dict[str, Tensor],
        fn: Callable[[dict[str, Tensor]], T],
    ) -> T:
        if not self.enabled:
            return fn(inputs)

        full_key = (
            key,
            tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(inputs.items())),
        )
        recorded = self._graphs.get(full_key)
        if recorded is not None:
            if self._hparams() != recorded.hparams:
                self._disable(
                    "an optimizer setting other than lr changed (e.g. a scheduler that "
                    "cycles momentum), and the recorded step can't follow it"
                )
                return fn(inputs)
            for k, v in inputs.items():
                recorded.inputs[k].copy_(v)
            self._load_lrs()
            recorded.graph.replay()
            return recorded.outputs

        self._calls[full_key] += 1
        if len(self._graphs) >= self.max_graphs:
            if not self._warned_shapes:
                self._warned_shapes = True
                get_root_logger().warning(
                    "CUDA graph: input shapes keep changing; steps with new shapes "
                    "run without a graph."
                )
            return fn(inputs)

        # Warmup and recording read the same input tensors, so the step sees the same
        # memory layout throughout: torch.compile would otherwise recompile for the
        # new layout while recording, which it can't do.
        static_inputs = self._inputs.get(full_key)
        if static_inputs is None:
            static_inputs = {k: torch.empty_like(v) for k, v in inputs.items()}
            self._inputs[full_key] = static_inputs
        for k, v in inputs.items():
            static_inputs[k].copy_(v)

        if self._calls[full_key] <= self.warmup_steps:
            # Warmup runs on a side stream, as the CUDA graph docs require.
            current = torch.cuda.current_stream()
            self._side_stream.wait_stream(current)
            with torch.cuda.stream(self._side_stream):
                out = fn(static_inputs)
            current.wait_stream(self._side_stream)
            return out

        try:
            return self._record(full_key, static_inputs, fn)
        except Exception as e:  # noqa: BLE001
            self._disable(f"recording failed: {str(e).strip().splitlines()[0]}")
            return fn(inputs)

    def _record(
        self,
        key: Hashable,
        static_inputs: dict[str, Tensor],
        fn: Callable[[dict[str, Tensor]], T],
    ) -> T:
        torch.cuda.synchronize()
        # Release what the warmup steps cached first, so it doesn't sit next to the
        # graph's own memory pool and raise peak VRAM.
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        saved_lrs = self._swap_lrs_in()
        try:
            # thread_local: the DataLoader pin-memory thread allocates pinned memory
            # meanwhile, which the default global mode forbids during a recording
            with (
                warnings.catch_warnings(),
                torch.cuda.graph(
                    graph, pool=self._pool, capture_error_mode="thread_local"
                ),
            ):
                # Autograd nodes from the warmup steps ran on the side stream, so
                # recording syncs once with it. That's harmless here.
                warnings.filterwarnings(
                    "ignore", message="The AccumulateGrad node's stream does not match"
                )
                outputs = fn(static_inputs)
        finally:
            self._swap_lrs_out(saved_lrs)
        self._pool = graph.pool()
        self._graphs[key] = _Graph(graph, static_inputs, outputs, self._hparams())
        get_root_logger().info(
            "CUDA graph: recorded the training step (%d graph%s).",
            len(self._graphs),
            "" if len(self._graphs) == 1 else "s",
        )
        # recording runs nothing, so replay it for this step
        self._load_lrs()
        graph.replay()
        return outputs

    def _disable(self, reason: str) -> None:
        get_root_logger().warning(
            "CUDA graph disabled, training continues without it: %s", reason
        )
        self.enabled = False
        self._graphs.clear()
        self._inputs.clear()
        self._pool = None
        # a failed recording leaves the scalers and optimizers mid-step
        for scaler in self.scalers:
            scaler._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)  # noqa: SLF001
        for optimizer in self.optimizers:
            for attr in ("grad_scale", "found_inf"):
                if hasattr(optimizer, attr):
                    delattr(optimizer, attr)
        _clear_capture_error()
        torch.cuda.empty_cache()

    def _hparams(self) -> list[tuple[Any, ...]]:
        return [
            tuple(g.get(h) for h in _FIXED_HPARAMS)
            for o in self.optimizers
            for g in o.param_groups
        ]

    def _swap_lrs_in(self) -> list[list[Any]]:
        saved = []
        for o, tensors in zip(self.optimizers, self._lr_tensors, strict=True):
            saved.append([g["lr"] for g in o.param_groups])
            for g, t in zip(o.param_groups, tensors, strict=True):
                t.fill_(float(g["lr"]))
                g["lr"] = t
        return saved

    def _swap_lrs_out(self, saved: list[list[Any]]) -> None:
        for o, lrs in zip(self.optimizers, saved, strict=True):
            for g, lr in zip(o.param_groups, lrs, strict=True):
                g["lr"] = lr

    def _load_lrs(self) -> None:
        for o, tensors in zip(self.optimizers, self._lr_tensors, strict=True):
            for g, t in zip(o.param_groups, tensors, strict=True):
                t.fill_(float(g["lr"]))
