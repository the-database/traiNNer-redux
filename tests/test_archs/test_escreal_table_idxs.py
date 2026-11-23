import pytest
import torch
from traiNNer.archs.escreal_arch import WindowAttention


def _reference(window_size: int, heads: int) -> torch.Tensor:
    # the original per-element loop
    idxs = []
    for head in range(heads):
        for h in range(window_size**2):
            for w in range(window_size**2):
                q_h, q_w = h // window_size, h % window_size
                k_h, k_w = w // window_size, w % window_size
                rel_h = k_h - q_h + window_size - 1
                rel_w = k_w - q_w + window_size - 1
                idxs.append((head, rel_h * (2 * window_size - 1) + rel_w))
    return torch.tensor(idxs, dtype=torch.long)


@pytest.mark.parametrize(("window_size", "heads"), [(1, 1), (3, 2), (8, 4)])
def test_table_idxs_match_reference(window_size: int, heads: int) -> None:
    assert torch.equal(
        WindowAttention.create_table_idxs(window_size, heads),
        _reference(window_size, heads),
    )
