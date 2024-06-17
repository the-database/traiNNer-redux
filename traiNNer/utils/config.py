import argparse
from collections.abc import Mapping
from typing import Any

from traiNNer.utils.options import parse_options


class Config:
    _config = None
    _args = None

    @classmethod
    def load_config(
        cls, root_path: str, is_train: bool = True
    ) -> tuple[Mapping[str, Any], argparse.Namespace]:
        if cls._config is None:
            cls._config, cls._args = parse_options(root_path, is_train)

        return cls._config, cls._args

    @classmethod
    def get_config(cls) -> tuple[Mapping[str, Any], argparse.Namespace]:
        if cls._config is None:
            raise RuntimeError("Config has not been loaded. Call load_config first.")

        return cls._config, cls._args

    @classmethod
    def get_scale(cls) -> int:
        return cls.get_config()[0]["scale"]