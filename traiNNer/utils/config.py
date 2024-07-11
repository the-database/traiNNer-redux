import argparse

from traiNNer.utils.options import parse_options
from traiNNer.utils.redux_options import ReduxOptions


class Config:
    _config = None
    _args = None

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._config is not None

    @classmethod
    def load_config_from_file(
        cls, root_path: str, is_train: bool = True
    ) -> tuple[ReduxOptions, argparse.Namespace]:
        if cls._config is None:
            cls._config, cls._args = parse_options(root_path, is_train)

        assert cls._args is not None
        return cls._config, cls._args

    @classmethod
    def get_config(cls) -> tuple[ReduxOptions, argparse.Namespace | None]:
        if cls._config is None:
            raise RuntimeError("Config has not been loaded. Call load_config first.")

        return cls._config, cls._args

    @classmethod
    def get_scale(cls) -> int:
        return cls.get_config()[0].scale

    @classmethod
    def get_manual_seed(cls) -> int | None:
        return cls.get_config()[0].manual_seed
