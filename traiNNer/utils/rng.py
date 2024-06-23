import numpy as np
from numpy.random import Generator
from traiNNer.utils.config import Config


class RNG:
    _rng = None

    @classmethod
    def init_rng(cls, seed: int) -> None:
        cls._rng = np.random.default_rng(seed)

    @classmethod
    def get_rng(cls) -> Generator:
        if cls._rng is None:
            seed = Config.get_manual_seed()
            if seed is None:
                raise RuntimeError("Manual seed is not set.")
            cls._rng = np.random.default_rng(seed)
        return cls._rng
