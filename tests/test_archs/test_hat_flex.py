import copy

import pytest
import torch
from torch import Tensor
from traiNNer.archs.hat_arch import HAT, hat_s

HAS_FLEX = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def small_hat(window_size: int) -> HAT:
    torch.manual_seed(0)
    net = HAT(
        upscale=2,
        embed_dim=60,
        depths=(2, 2),
        num_heads=(6, 6),
        window_size=window_size,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
    )
    with torch.no_grad():  # trained-looking bias tables, so the bias path matters
        for m in net.modules():
            table = getattr(m, "relative_position_bias_table", None)
            if isinstance(table, torch.Tensor):
                table.normal_(0, 0.5)
    return net


def rel(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / b.norm())


@pytest.mark.skipif(not HAS_FLEX, reason="needs a CUDA GPU (sm_80+)")
@pytest.mark.parametrize(
    ("window_size", "size"), [(8, 24), (16, 32), (16, 48), (32, 64)]
)
def test_flex_matches_stock(window_size: int, size: int) -> None:
    """FlexAttention computes the stock attention: outputs and every gradient, the bias tables'
    included (window 8 and 16 cover the padding to 128-token blocks, the OCA's negative indices)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stock = small_hat(window_size).cuda().train()
    stock.attn_impl = "stock"
    flex = copy.deepcopy(stock)
    flex.attn_impl = "flex"
    torch.manual_seed(1)
    x = torch.rand(2, 3, size, size, device="cuda")
    target = torch.rand(2, 3, 2 * size, 2 * size, device="cuda")
    ys, yf = stock(x), flex(x)
    (ys - target).abs().mean().backward()
    (yf - target).abs().mean().backward()
    assert rel(yf, ys) < 1e-5
    grads = dict(stock.named_parameters())
    for name, p in flex.named_parameters():
        g = grads[name].grad
        assert p.grad is not None and g is not None, name
        if g.norm() > 1e-6:
            assert rel(p.grad, g) < 1e-3, name


def test_auto_uses_flex_only_while_training() -> None:
    net = small_hat(16)
    x = torch.rand(1, 3, 32, 32)
    assert not net.eval().use_flex(x)
    assert not net.train().use_flex(x)  # CPU tensor
    if HAS_FLEX:
        assert net.cuda().train().use_flex(x.cuda())
        assert not net.eval().use_flex(x.cuda())
        net.attn_impl = "stock"
        assert not net.train().use_flex(x.cuda())


def test_attn_impl_stays_out_of_checkpoints() -> None:
    net = hat_s(scale=2, attn_impl="stock")
    assert net.attn_impl == "stock"
    assert "attn_impl" not in net.hyperparameters
    assert set(net.state_dict()) == set(hat_s(scale=2).state_dict())
    with pytest.raises(ValueError):
        hat_s(scale=2, attn_impl="fast")  # pyright: ignore[reportArgumentType]
