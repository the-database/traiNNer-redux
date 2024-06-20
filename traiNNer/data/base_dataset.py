from abc import abstractmethod
from typing import Any

from torch.utils import data
from traiNNer.utils.types import DataFeed

from ..utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BaseDataset(data.Dataset):
    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__()
        self.opt = opt

    @abstractmethod
    def __getitem__(self, index: int) -> DataFeed:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
