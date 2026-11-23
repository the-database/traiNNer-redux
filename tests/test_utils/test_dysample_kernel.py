"""DySample's Triton sampling kernel must match F.grid_sample(bilinear, border,
align_corners=False) on channel groups, forward and backward."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs CUDA",
)


def _reference(x: torch.Tensor, grid: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    out = F.grid_sample(
        x.float().contiguous().view(b * groups, -1, h, w),
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return out.view(b, c, grid.shape[1], grid.shape[2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize(("c", "h", "w"), [(64, 16, 16), (24, 9, 13), (128, 8, 8)])
def test_matches_grid_sample(
    dtype: torch.dtype, channels_last: bool, c: int, h: int, w: int
) -> None:
    from traiNNer.ops.dysample_sample import dysample_sample

    torch.manual_seed(0)
    b, groups = 2, 4
    oh, ow = 2 * h, 2 * w
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    x0 = torch.randn(b, c, h, w, device="cuda").to(dtype).contiguous(memory_format=fmt)
    base = torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, ow, device="cuda"),
            torch.linspace(-1, 1, oh, device="cuda"),
            indexing="xy",
        ),
        -1,
    )
    # some points land past the border, where border padding clips them
    grid0 = (
        base[None] + 0.05 * torch.randn(b * groups, oh, ow, 2, device="cuda")
    ) * 1.05
    gout = torch.randn(b, c, oh, ow, device="cuda")

    results = []
    for fn in ("ref", "kernel"):
        x = x0.clone().requires_grad_(True)
        grid = grid0.clone().requires_grad_(True)
        if fn == "ref":
            out = _reference(x, grid, groups)
        else:
            out = dysample_sample(x, grid, groups, torch.float32, fmt)
        out.backward(gout)
        assert x.grad is not None and grid.grad is not None
        results.append((out, x.grad.float(), grid.grad))

    (ro, rgx, rgg), (ko, kgx, kgg) = results
    assert ko.shape == ro.shape
    torch.testing.assert_close(ko, ro, rtol=1e-4, atol=1e-4)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(kgx, rgx, rtol=tol, atol=tol)
    torch.testing.assert_close(kgg, rgg, rtol=1e-3, atol=1e-3)


def test_output_dtype_and_layout() -> None:
    from traiNNer.ops.dysample_sample import dysample_sample

    x = torch.randn(1, 8, 4, 4, device="cuda", dtype=torch.float16).contiguous(
        memory_format=torch.channels_last
    )
    grid = torch.zeros(4, 8, 8, 2, device="cuda")
    out = dysample_sample(x, grid, 4, torch.float16, torch.channels_last)
    assert out.dtype == torch.float16
    assert out.is_contiguous(memory_format=torch.channels_last)
