from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from traiNNer.archs.conv3xc import Conv3XC
from traiNNer.archs.spanf3_arch import spanf3


def test_fusion_matches_branch_gradients_update_and_saved_deploy_kernel() -> None:
    torch.manual_seed(6)
    reference = Conv3XC(3, 5, gain1=2, s=2, relu=True).double()
    fused = deepcopy(reference)
    fused.fused_training = True
    image = torch.randn(2, 3, 7, 12, dtype=torch.float64)
    weights = torch.randn(2, 5, 4, 6, dtype=torch.float64)
    updates = []
    for block in (reference, fused):
        x = image.clone().requires_grad_()
        y = block(x)
        (y * weights).sum().backward()
        assert x.grad is not None
        grads = [x.grad]
        for parameter in block.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                grads.append(parameter.grad)
        torch.optim.AdamW(block.parameters(), lr=1e-3).step()
        updates.append((y.detach(), grads, list(block.parameters())))
    for before, after in zip(*updates, strict=True):
        torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-12)

    # Save directly in training mode, as train.py does between validations.
    state = fused.state_dict()
    deployed = F.conv2d(
        image, state["eval_conv.weight"], state["eval_conv.bias"], stride=2, padding=1
    )
    torch.testing.assert_close(F.leaky_relu(deployed, 0.05), reference(image))
    fused.switch_to_deploy()
    torch.testing.assert_close(fused.eval()(image), reference(image))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_spanf3_amp_forward_and_backward(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device lacks bf16")
    torch.manual_seed(17)
    reference = (
        spanf3(scale=2, num_blocks=2, feature_channels=16)
        .cuda()
        .to(memory_format=torch.channels_last)
    )
    fused = (
        spanf3(scale=2, num_blocks=2, feature_channels=16, fused_training=True)
        .cuda()
        .to(memory_format=torch.channels_last)
    )
    fused.load_state_dict(reference.state_dict(), strict=True)
    # Odd dimensions exercise reflect padding, CRD unshuffle and the output crop.
    x = torch.randn(2, 3, 19, 25, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    target = torch.randn(2, 3, 38, 50, device="cuda")
    results = []
    for net in (reference, fused):
        with torch.autocast("cuda", dtype=dtype):
            y = net(x)
            loss = (y.float() * target).sum()
        loss.backward()
        grads = [
            p.grad.flatten()
            for p in net.parameters()
            if p.requires_grad and p.grad is not None
        ]
        results.append((y.float(), torch.cat(grads)))
    # Individual ReLU decisions near zero are sensitive to rounding. Check the
    # forward signal and gradient separately, with tighter bounds on the output.
    limits = (0.003, 0.012) if dtype == torch.float16 else (0.015, 0.03)
    for (branch, composed), limit in zip(
        zip(*results, strict=True), limits, strict=True
    ):
        assert branch.isfinite().all() and composed.isfinite().all()
        relative_error = (branch - composed).norm() / branch.norm()
        assert relative_error < limit
