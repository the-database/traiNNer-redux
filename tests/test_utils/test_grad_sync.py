"""GradSync must do DistributedDataParallel's job: rank 0's weights everywhere at the
start, gradients averaged across the ranks, and rank 0's changing buffers each step."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

WORLD = 2


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.sn = spectral_norm(nn.Conv2d(8, 4, 3, padding=1))
        self.constant: Tensor
        self.register_buffer("constant", torch.randn(16))  # never changes

    def forward(self, x: Tensor) -> Tensor:
        return self.sn(torch.relu(self.bn(self.conv(x)))) + self.constant.sum() * 0


def gathered(t: Tensor) -> list[Tensor]:
    out = [torch.empty_like(t) for _ in range(WORLD)]
    dist.all_gather(out, t.contiguous())
    return out


def worker(rank: int, init_file: str) -> None:
    from traiNNer.utils.grad_sync import GradSync

    dist.init_process_group(
        "gloo",
        init_method="file:///" + init_file.lstrip("/"),
        rank=rank,
        world_size=WORLD,
    )
    try:
        torch.manual_seed(rank)  # each rank initialises differently, as in training
        net = Net().to(memory_format=torch.channels_last)
        sync = GradSync([net])
        for t in list(net.parameters()) + list(net.buffers()):
            a, b = gathered(t.detach())
            assert torch.equal(a, b), "the start broadcast"

        snap = net.constant.clone()
        for step in range(3):
            sync.sync_buffers(net)
            if step == 2:  # the changing buffers follow rank 0 from the second step
                for name in ("bn.running_mean", "bn.running_var", "sn.weight_u"):
                    a, b = gathered(net.get_buffer(name))
                    assert torch.equal(a, b), name
            torch.manual_seed(100 + rank)  # different data on each rank
            x = torch.rand(2, 3, 8, 8).to(memory_format=torch.channels_last)
            net(x).square().mean().backward()
            local = [p.grad.clone() for p in net.parameters()]  # pyright: ignore[reportOptionalMemberAccess]
            sync.all_reduce_grads(net)
            for p, g in zip(net.parameters(), local, strict=True):
                expect = torch.stack(gathered(g)).mean(0)
                assert p.grad is not None
                torch.testing.assert_close(p.grad, expect)
                a, b = gathered(p.grad)
                assert torch.equal(a, b)
            with torch.no_grad():
                for p in net.parameters():
                    assert p.grad is not None
                    p.sub_(0.1 * p.grad)
                    p.grad.zero_()

        kept = {
            name
            for names, _, _ in sync._nets[id(net)].buffers  # noqa: SLF001
            for _, name in names
        }
        assert {"running_mean", "running_var", "weight_u"} <= kept
        assert "constant" not in kept
        assert torch.equal(net.constant, snap)
    finally:
        dist.destroy_process_group()


def test_grad_sync_matches_ddp_semantics() -> None:
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init").replace("\\", "/")
        mp.spawn(worker, args=(init_file,), nprocs=WORLD, join=True)  # pyright: ignore[reportPrivateImportUsage]
