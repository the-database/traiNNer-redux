"""Training through CudaGraphStep must match eager training step for step."""

import threading
from functools import partial

import pytest
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from traiNNer.optimizers import build_optimizer
from traiNNer.utils.amp_util import (
    guard_nonfinite,
    make_grad_scaler,
    update_grad_scaler,
)
from traiNNer.utils.cuda_graph import (
    CudaGraphStep,
    move_tensor_attributes,
    supports_cuda_graph,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

STEPS = 16
NAN_STEP = 7


def train(
    use_graph: bool,
    accum: int,
    amp: torch.dtype | None,
    fail: bool = False,
) -> tuple[Tensor, list[float], CudaGraphStep | None]:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 3, 3, padding=1)
    ).cuda()
    opt = AdamW(
        net.parameters(), lr=1e-3, weight_decay=0.01, fused=True, capturable=True
    )
    # the lr changes while graphs are replayed
    sched = StepLR(opt, step_size=5, gamma=0.5)
    loss_scaling = amp == torch.float16
    scaler = make_grad_scaler(loss_scaling)
    runner = CudaGraphStep([opt], [scaler]) if use_graph else None

    gen = torch.Generator().manual_seed(1)
    batches = [
        torch.rand(4, 3, 16, 16, generator=gen).cuda() for _ in range(STEPS * accum)
    ]

    def step(inputs: dict[str, Tensor], apply: bool) -> dict[str, Tensor]:
        with torch.autocast(
            "cuda", dtype=amp or torch.float32, enabled=amp is not None
        ):
            out = net(inputs["x"])
            loss = (out.float() - inputs["x"]).abs().mean() * inputs["w"] / accum
        backward_loss, _ = guard_nonfinite(loss)
        if fail:
            loss.item()  # a sync can't be recorded
        scaler.scale(backward_loss).backward()
        if apply:
            scaler.unscale_(opt)
            norm = get_total_norm(
                [p.grad for p in net.parameters() if p.grad is not None], foreach=True
            )
            clip_grads_with_norm_(net.parameters(), 0.5, norm, foreach=True)
            scaler.step(opt)
            update_grad_scaler(scaler, loss_scaling)
            opt.zero_grad(set_to_none=False)
        return {"loss": loss.detach()}

    losses = []
    i = 0
    for it in range(STEPS):
        for a in range(accum):
            apply = a == accum - 1
            weight = float("nan") if it == NAN_STEP else 1.0
            inputs = {"x": batches[i], "w": torch.tensor(weight, device="cuda")}
            i += 1
            fn = partial(step, apply=apply)
            out = runner(apply, inputs, fn) if runner is not None else fn(inputs)
            losses.append(float(out["loss"]))
        sched.step()
    params = torch.cat([p.detach().flatten() for p in net.parameters()])
    return params, losses, runner


@pytest.mark.parametrize("accum", [1, 3])
@pytest.mark.parametrize("amp", [None, torch.bfloat16, torch.float16])
def test_graph_matches_eager(accum: int, amp: torch.dtype | None) -> None:
    eager_params, eager_losses, _ = train(False, accum, amp)
    graph_params, graph_losses, runner = train(True, accum, amp)
    assert runner is not None
    assert runner.enabled
    # one graph for the step that applies the gradient, one for accumulating
    assert len(runner._graphs) == (1 if accum == 1 else 2)  # noqa: SLF001
    torch.testing.assert_close(graph_params, eager_params, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        graph_losses, eager_losses, rtol=0, atol=1e-6, equal_nan=True
    )


def test_failed_recording_falls_back_to_eager() -> None:
    eager_params, _, _ = train(False, 1, torch.bfloat16, fail=True)
    graph_params, _, runner = train(True, 1, torch.bfloat16, fail=True)
    assert runner is not None
    assert not runner.enabled
    torch.testing.assert_close(graph_params, eager_params, rtol=0, atol=0)


def test_optimizer_support() -> None:
    param = nn.Parameter(torch.zeros(4, device="cuda"))
    opt = build_optimizer([param], {"type": "AdamW", "lr": 1e-3}, capturable=True)
    assert supports_cuda_graph(opt) is None
    opt = build_optimizer([param], {"type": "AdamW", "lr": 1e-3})
    assert supports_cuda_graph(opt) is not None
    opt = build_optimizer(
        [param], {"type": "AdamW", "lr": 1e-3, "foreach": True}, capturable=True
    )
    assert supports_cuda_graph(opt) is not None


def test_move_tensor_attributes() -> None:
    net = nn.Sequential(nn.Conv2d(3, 3, 1))
    net[0].mean = torch.ones(1, 3, 1, 1)  # pyright: ignore[reportArgumentType]
    net = net.cuda()
    move_tensor_attributes(net, torch.device("cuda"))
    assert net[0].mean.is_cuda  # pyright: ignore[reportAttributeAccessIssue]
    assert "0.mean" not in net.state_dict()


def test_recording_with_pinned_allocs_in_another_thread() -> None:
    """The DataLoader pin-memory thread allocates pinned memory while the step is
    recorded. That must neither fail the recording nor crash the other thread."""
    stop = threading.Event()
    errors: list[BaseException] = []

    def pin_loop() -> None:
        try:
            while not stop.is_set():
                torch.empty(1 << 16, pin_memory=True)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    thread = threading.Thread(target=pin_loop)
    thread.start()
    try:
        _, _, runner = train(True, 1, torch.bfloat16)
    finally:
        stop.set()
        thread.join()
    assert runner is not None
    assert runner.enabled
    assert not errors
