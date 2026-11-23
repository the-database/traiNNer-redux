"""DataLoader worker setup: thread environment, parallel spawn on Windows, and the
torchvision-free image helpers the workers use."""

import os
import signal
import sys
from functools import partial

import pytest
import torch
from torch.utils.data import Dataset
from traiNNer.data import worker_init_fn
from traiNNer.data.parallel_spawn import parallel_spawn_context
from traiNNer.data.prefetch_dataloader import (
    WORKER_THREAD_ENV,
    DataLoader,
    worker_thread_env,
)
from traiNNer.utils.img_util import normalize_, rgb_to_grayscale
from traiNNer.utils.win_qos import disable_power_throttling


class IndexDataset(Dataset):
    """Returns its index and the worker's thread environment."""

    def __len__(self) -> int:
        return 64

    def __getitem__(self, i: int) -> dict:
        return {
            "i": torch.tensor(i),
            "omp": os.environ.get("OMP_NUM_THREADS", ""),
            "threads": torch.tensor(torch.get_num_threads()),
            "ignores_interrupt": signal.getsignal(signal.SIGINT) == signal.SIG_IGN,
        }


def test_normalize_matches_torchvision() -> None:
    from torchvision.transforms.functional import normalize

    x = torch.rand(3, 8, 8)
    mean, std = [0.4, 0.5, 0.6], [0.2, 0.25, 0.3]
    torch.testing.assert_close(
        normalize_(x.clone(), mean, std), normalize(x, mean, std)
    )


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shape", [(3, 8, 8), (2, 3, 8, 8), (1, 8, 8)])
def test_rgb_to_grayscale_matches_torchvision(
    shape: tuple[int, ...], channels: int
) -> None:
    from torchvision.transforms.functional import rgb_to_grayscale as tv_gray

    x = torch.rand(shape)
    torch.testing.assert_close(
        rgb_to_grayscale(x, num_output_channels=channels),
        tv_gray(x, num_output_channels=channels),
    )


def test_worker_thread_env_is_temporary(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in WORKER_THREAD_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")  # a user's own setting is kept
    with worker_thread_env():
        assert os.environ["OMP_NUM_THREADS"] == "7"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_workers_get_one_thread_and_all_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for k in WORKER_THREAD_ENV:
        monkeypatch.delenv(k, raising=False)
    loader = DataLoader(
        IndexDataset(),
        batch_size=4,
        num_workers=4,
        multiprocessing_context=parallel_spawn_context(),
        worker_init_fn=partial(worker_init_fn, num_workers=4, rank=0, seed=None),
    )
    batches = list(loader)
    assert torch.cat([b["i"] for b in batches]).tolist() == list(range(64))
    assert all(o == "1" for b in batches for o in b["omp"])
    assert all(int(t) == 1 for b in batches for t in b["threads"])
    assert all(b["ignores_interrupt"].all() for b in batches)
    # the parent keeps its own environment
    assert "OMP_NUM_THREADS" not in os.environ


def test_parallel_spawn_only_on_windows() -> None:
    ctx = parallel_spawn_context()
    if sys.platform == "win32":
        assert ctx is not None
        assert ctx.get_start_method() == "spawn"
    else:
        assert ctx is None


def test_disable_power_throttling() -> None:
    assert disable_power_throttling() == (sys.platform == "win32")
