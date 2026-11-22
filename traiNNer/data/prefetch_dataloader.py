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


def _prefetch(source: Iterator, depth: int) -> Iterator:
    """Read ahead in order and surface decoding errors on the consumer thread."""
    pending = queue.Queue(depth)
    stopped = threading.Event()
    end = object()
    error = None

    def produce() -> None:
        nonlocal error
        try:
            while not stopped.is_set():
                pending.put(next(source))
        except StopIteration:
            pass
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer thread
            error = exc
        finally:
            if not stopped.is_set():
                pending.put(end)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    try:
        while (item := pending.get()) is not end:
            yield item
        if error is not None:
            raise error
    finally:
        stopped.set()
        # Release a producer blocked on a full queue when training ends mid-epoch.
        try:
            pending.get_nowait()
        except queue.Empty:
            pass
        worker.join()


class PrefetchDataLoader(DataLoader):
    def __init__(self, num_prefetch_queue: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_prefetch_queue = num_prefetch_queue

    def __iter__(self) -> Iterator:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _prefetch(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher:
    def __init__(self, loader: TorchDataLoader) -> None:
        self.ori_loader = loader
        self.loader: Iterator | None = None

    def __iter__(self) -> Iterator:
        return iter(self.ori_loader)

    def next(self) -> Any:
        if self.loader is None:
            self.reset()
        assert self.loader is not None
        return next(self.loader, None)

    def reset(self) -> None:
        self.loader = iter(self)


class CUDAPrefetcher(CPUPrefetcher):
    """Queue the next transfer while the previous batch executes on the GPU."""

    def __init__(self, loader: TorchDataLoader, opt: ReduxOptions) -> None:
        super().__init__(loader)
        self.stream = torch.cuda.Stream()
        self.memory_format = (
            torch.channels_last
            if opt.use_amp and opt.use_channels_last
            else torch.preserve_format
        )

    def __iter__(self) -> Iterator:
        for batch in self.ori_loader:
            with torch.cuda.stream(self.stream):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(
                            device=self.stream.device,
                            memory_format=self.memory_format
                            if value.ndim == 4
                            else torch.preserve_format,
                            non_blocking=True,
                        )
            consumer = torch.cuda.current_stream(self.stream.device)
            consumer.wait_stream(self.stream)
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(consumer)
            yield batch
