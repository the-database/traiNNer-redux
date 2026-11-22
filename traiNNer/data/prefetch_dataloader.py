import os
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from traiNNer.utils.redux_options import ReduxOptions

# Read by the libraries in each worker process when they load. The workers already
# run in parallel, so one thread each is enough, and OpenBLAS also reserves memory for
# every thread: on Windows, where workers are fresh processes and memory isn't
# overcommitted, that took a worker from 1.1 GB to 2.5 GB of committed memory.
WORKER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VIPS_CONCURRENCY": "1",
}


@contextmanager
def worker_thread_env() -> Iterator[None]:
    """Set WORKER_THREAD_ENV while worker processes start, leaving values the user
    set alone, so this process keeps its own thread pools."""
    added = [k for k in WORKER_THREAD_ENV if k not in os.environ]
    for k in added:
        os.environ[k] = WORKER_THREAD_ENV[k]
    try:
        yield
    finally:
        for k in added:
            os.environ.pop(k, None)


class DataLoader(TorchDataLoader):
    """DataLoader whose worker processes start with one thread per library."""

    def _get_iterator(self) -> Any:
        if self.num_workers == 0:
            return super()._get_iterator()
        with worker_thread_env():
            return super()._get_iterator()


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Reference: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator: TorchDataLoader, num_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self) -> Any:
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self) -> Iterator:
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Reference: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue: int, **kwargs) -> None:
        self.num_prefetch_queue = num_prefetch_queue
        super().__init__(**kwargs)

    def __iter__(self) -> PrefetchGenerator:  # type: ignore
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)  # type: ignore


class CPUPrefetcher:
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader: TorchDataLoader) -> None:
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self) -> Any:
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self) -> None:
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher:
    """CUDA prefetcher.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader: TorchDataLoader, opt: ReduxOptions) -> None:
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device("cuda" if opt.num_gpu != 0 else "cpu")
        self.batch = None
        self.memory_format = (
            torch.channels_last
            if opt.use_amp and opt.use_channels_last
            else torch.preserve_format
        )
        self.preload()

    def preload(self) -> None:
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return
        # put tensors to gpu
        with torch.cuda.stream(self.stream):  # type: ignore
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    # channels_last only applies to 4D image batches, not to e.g.
                    # the 3D blur kernels of OTF datasets or 5D video clips
                    self.batch[k] = v.to(
                        device=self.device,
                        memory_format=self.memory_format
                        if v.dim() == 4
                        else torch.preserve_format,
                        non_blocking=True,
                    )

    def next(self) -> Any:
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            # The batch was allocated on self.stream but is consumed on the current
            # stream. Without record_stream, the caching allocator could hand its
            # memory to the next preload while the current stream is still reading it.
            for v in batch.values():
                if torch.is_tensor(v) and v.is_cuda:
                    v.record_stream(current_stream)
        self.preload()
        return batch

    def reset(self) -> None:
        self.loader = iter(self.ori_loader)
        self.preload()
