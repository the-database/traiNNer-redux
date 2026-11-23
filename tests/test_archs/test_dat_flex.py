import copy

import pytest
import torch
from torch import Tensor
from traiNNer.archs.dat_arch import DAT, dat_2

HAS_FLEX = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def small_dat(split_size: tuple[int, int]) -> DAT:
    # depth 4 + 2: group 0's block 2 and group 1's block 0 are the shifted spatial blocks
    torch.manual_seed(0)
    return DAT(
        upscale=2,
        embed_dim=60,
        depth=(4, 2),
        num_heads=(6, 6),
        split_size=split_size,
        expansion_factor=2.0,
        drop_path_rate=0.0,
    )


def rel(a: Tensor, b: Tensor) -> float:
    return float((a - b).detach().norm() / b.detach().norm())


@pytest.mark.skipif(not HAS_FLEX, reason="needs a CUDA GPU (sm_80+)")
@pytest.mark.parametrize(
    ("split_size", "size"), [((4, 8), 32), ((8, 8), 32), ((4, 8), 36)]
)
def test_flex_matches_stock(split_size: tuple[int, int], size: int) -> None:
    """FlexAttention computes the stock spatial attention: outputs and every gradient, the
    dynamic position bias MLP's included, at least as close to a float64 run as the stock path
    (rectangular and square windows, the shifted blocks, and a size the windows don't divide)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = small_dat(split_size).cuda().train()
    nets = {}
    for name, impl, dtype in (
        ("stock", "stock", torch.float32),
        ("flex", "flex", torch.float32),
        ("ref", "stock", torch.float64),
    ):
        nets[name] = copy.deepcopy(base).to(dtype)
        nets[name].attn_impl = impl
    torch.manual_seed(1)
    x = torch.rand(2, 3, size, size, device="cuda")
    target = torch.rand(2, 3, 2 * size, 2 * size, device="cuda")
    outs = {}
    for name, net in nets.items():
        dtype = next(net.parameters()).dtype
        outs[name] = net(x.to(dtype))
        (outs[name] - target.to(dtype)).abs().mean().backward()
    assert rel(outs["flex"].double(), outs["ref"]) < 1e-5
    params = {name: dict(net.named_parameters()) for name, net in nets.items()}
    for name, p64 in params["ref"].items():
        g64 = p64.grad
        assert g64 is not None, name
        if g64.norm() < 1e-9:
            continue
        gs, gf = params["stock"][name].grad, params["flex"][name].grad
        assert gs is not None and gf is not None, name
        # 1e-4: the float32 noise floor of the smallest gradients
        assert rel(gf.double(), g64) < 1.5 * rel(gs.double(), g64) + 1e-4, name


def test_auto_uses_flex_only_while_training() -> None:
    net = small_dat((4, 8))
    x = torch.rand(1, 3, 32, 32)
    assert not net.eval().use_flex(x)
    assert not net.train().use_flex(x)  # CPU tensor
    if HAS_FLEX:
        assert net.cuda().train().use_flex(x.cuda())
        assert not net.eval().use_flex(x.cuda())
        net.attn_impl = "stock"
        assert not net.train().use_flex(x.cuda())


def test_attn_impl_stays_out_of_checkpoints() -> None:
    net = dat_2(scale=2, attn_impl="stock")
    assert net.attn_impl == "stock"
    assert "attn_impl" not in net.hyperparameters
    assert set(net.state_dict()) == set(dat_2(scale=2).state_dict())
    with pytest.raises(ValueError):
        dat_2(scale=2, attn_impl="fast")  # pyright: ignore[reportArgumentType]
