from contextlib import closing
from itertools import islice

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.prefetch_dataloader import PrefetchDataLoader
from traiNNer.training import training_batches
from traiNNer.utils.types import DataFeed


class _Images(Dataset):
    def __len__(self) -> int:
        return 17

    def __getitem__(self, index: int) -> DataFeed:
        return {"lq": torch.tensor(index)}


def test_sampler_resume_skips_decoding_and_carries_across_epochs() -> None:
    data = _Images()
    sampler = EnlargedSampler(data, num_replicas=2, rank=1)
    loader = DataLoader(data, batch_size=2, sampler=sampler, drop_last=True)

    def read(epoch: int, offset: int, count: int) -> list[tuple[int, int, list[int]]]:
        result = []
        with closing(training_batches(loader, sampler, 2, epoch, offset)) as batches:
            for e, n, batch in islice(batches, count):
                assert "lq" in batch
                result.append((e, n, batch["lq"].tolist()))
        return result

    whole = read(0, 0, 6)
    # Match the upstream permutation and per-rank slice, including its padded sample.
    indices = torch.randperm(18, generator=torch.Generator().manual_seed(0))
    assert [v for _, _, values in whole[:4] for v in values] == (
        indices[1::2][:8] % len(data)
    ).tolist()
    # An in-epoch checkpoint and one exactly at the drop_last boundary.
    for prefix in (3, 4):
        epoch, consumed, _ = whole[prefix - 1]
        assert read(epoch, consumed, 6 - prefix) == whole[prefix:]


def test_cpu_prefetch_delivers_decode_error_to_training_loop() -> None:
    class Broken(_Images):
        def __getitem__(self, index: int) -> DataFeed:
            if index == 1:
                raise OSError("bad image")
            return super().__getitem__(index)

    loader = PrefetchDataLoader(1, dataset=Broken(), batch_size=1, num_workers=0)
    batches = iter(loader)
    assert next(batches)["lq"].item() == 0
    with pytest.raises(OSError, match="bad image"):
        next(batches)
