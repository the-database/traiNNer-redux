from abc import abstractmethod

from torch.utils import data

from traiNNer.utils.optionsfile import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class BaseDataset(data.Dataset):
    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__()
        self.opt = opt

    @abstractmethod
    def __getitem__(self, index: int) -> DataFeed:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
