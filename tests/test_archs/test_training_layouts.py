import copy
from typing import Any

import pytest
import torch
from spandrel.architectures.ATD import ATD as SPANDREL_ATD
from spandrel.architectures.DAT.__arch.DAT import DAT as SPANDREL_DAT
from torch import Tensor, nn
from traiNNer.archs.atd_arch import atd
from traiNNer.archs.dat_arch import DAT
from traiNNer.archs.hat_arch import HAT
from traiNNer.archs.srvgg_arch import compact
from traiNNer.archs.swinir_arch import swinir_m


def compare_gradients(reference: nn.Module, changed: nn.Module, x: Tensor) -> None:
    x1 = x.detach().requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    reference_out, changed_out = reference(x1), changed(x2)
    torch.testing.assert_close(changed_out, reference_out)
    upstream = torch.randn_like(reference_out)
    reference_out.backward(upstream)
    changed_out.backward(upstream)
    torch.testing.assert_close(x2.grad, x1.grad)
    ref_params = dict(reference.named_parameters())
    for name, p in changed.named_parameters():
        assert p.grad is not None and ref_params[name].grad is not None, name
        torch.testing.assert_close(p.grad, ref_params[name].grad, msg=name)


def test_dat_shifted_rectangles_match_spandrel() -> None:
    # Four blocks include unshifted, shifted and channel attention. Both axes need padding.
    opts: dict[str, Any] = {
        "img_size": 8,
        "embed_dim": 64,
        "split_size": [2, 4],
        "depth": [4],
        "num_heads": [4],
        "drop_path_rate": 0.0,
        "upscale": 2,
    }
    reference = SPANDREL_DAT(**opts).double()
    changed = DAT(**opts).double()
    changed.load_state_dict(reference.state_dict(), strict=True)
    compare_gradients(reference, changed, torch.rand(2, 3, 7, 11, dtype=torch.float64))


def test_hat_overlap_and_cached_mask_match_evaluation() -> None:
    reference = (
        HAT(
            upscale=2,
            img_size=16,
            embed_dim=60,
            depths=(2,),
            num_heads=(6,),
            window_size=4,
            drop_path_rate=0.0,
        )
        .double()
        .eval()
    )
    changed = copy.deepcopy(reference).train()
    for h, w in ((8, 12), (12, 8)):
        reference.zero_grad()
        changed.zero_grad()
        compare_gradients(
            reference, changed, torch.rand(2, 3, h, w, dtype=torch.float64)
        )


def test_atd_category_order_and_padding_match_spandrel() -> None:
    opts: dict[str, Any] = {
        "img_size": 8,
        "window_size": 4,
        "embed_dim": 32,
        "depths": (2,),
        "num_heads": (4,),
        "num_tokens": 8,
        "category_size": 17,
        "reducted_dim": 8,
        "upsampler": "pixelshuffle",
    }
    reference = SPANDREL_ATD(upscale=2, **opts).double()
    changed = atd(scale=2, **opts).double()
    changed.load_state_dict(reference.state_dict(), strict=True)
    compare_gradients(reference, changed, torch.rand(2, 3, 7, 11, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_fused_prelu_matches_native() -> None:
    reference = compact(num_feat=12, num_conv=2, scale=3).double().cuda()
    changed = (
        compact(num_feat=12, num_conv=2, scale=3, fused_training=True).double().cuda()
    )
    changed.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 3, 7, 11, device="cuda", dtype=torch.float64)
    compare_gradients(reference.eval(), changed.train(), x)
    torch.testing.assert_close(changed.eval()(x), reference(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("arch", ["atd", "swinir_m"])
def test_spandrel_graph_replays_on_new_pixels(arch: str) -> None:
    opts: dict[str, Any] = {
        "scale": 2,
        "img_size": 8,
        "window_size": 4,
        "embed_dim": 32,
        "depths": (2,),
        "num_heads": (4,),
    }
    net = (
        atd(**opts, num_tokens=8, category_size=16)
        if arch == "atd"
        else swinir_m(**opts, drop_path_rate=0.0)
    ).cuda()
    x = torch.randn(2, 3, 8, 12, device="cuda", requires_grad=True)

    def step() -> tuple[Tensor, Tensor]:
        output = net(x)
        return output, torch.autograd.grad(output.square().mean(), x)[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    with torch.no_grad():
        x.normal_()
    eager = step()
    graph.replay()
    for actual, expected in zip(captured, eager, strict=True):
        torch.testing.assert_close(actual, expected)
