from .options import parse_options


class Config:
    _config = None
    _args = None

    @classmethod
    def load_config(cls, root_path, is_train=True):
        if cls._config is None:
            cls._config, cls._args = parse_options(root_path, is_train)

        return cls._config, cls._args

    @classmethod
    def get_config(cls):
        if cls._config is None:
            raise RuntimeError("Config has not been loaded. Call load_config first.")

        return cls._config, cls._args

    @classmethod
    def get_scale(cls):
        return cls.get_config()[0]["scale"]
