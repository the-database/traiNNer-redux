"""traiNNer's DySampleV1 must match Spandrel's DySample: same state dict, same output."""

import pytest
import torch
from spandrel.architectures.__arch_helpers.dysample import DySample as SpandrelDySample
from traiNNer.archs.arch_util import DySampleV1

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("end_convolution", [False, True])
@pytest.mark.parametrize("hw", [(16, 16), (12, 20)])
def test_dysample_v1_matches_spandrel(
    device: str, end_convolution: bool, hw: tuple[int, int]
) -> None:
    torch.manual_seed(0)
    ref = SpandrelDySample(32, 3, 2, 4, end_convolution).to(device)
    ours = DySampleV1(32, 3, 2, 4, end_convolution).to(device)
    assert ref.state_dict().keys() == ours.state_dict().keys()
    with torch.no_grad():
        # nonzero scope so the learned offsets matter
        ref.scope.weight.normal_(0, 0.1)
    ours.load_state_dict(ref.state_dict())

    x = torch.rand(2, 32, *hw, device=device)
    x_ref = x.clone().requires_grad_()
    x_ours = x.clone().requires_grad_()
    out_ref = ref(x_ref)
    out_ours = ours(x_ours)
    torch.testing.assert_close(out_ours, out_ref, rtol=0, atol=0)

    grad = torch.rand_like(out_ref)
    out_ref.backward(grad)
    out_ours.backward(grad)
    assert x_ref.grad is not None and x_ours.grad is not None
    if device == "cpu":
        torch.testing.assert_close(x_ours.grad, x_ref.grad, rtol=0, atol=0)
    else:
        # grid_sample's CUDA backward accumulates with atomics, so its result
        # varies in the last bits from run to run
        torch.testing.assert_close(x_ours.grad, x_ref.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_dysample_trains_after_inference_mode(device: str) -> None:
    # the cached grid must not be an inference tensor when inference came first
    m = DySampleV1(32, 3, 2, 4, True).to(device)
    x = torch.rand(1, 32, 11, 13, device=device)
    with torch.inference_mode():
        m(x)
    m(x.requires_grad_()).sum().backward()
    assert x.grad is not None
