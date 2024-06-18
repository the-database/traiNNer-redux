import numpy as np
from traiNNer.utils.config import Config


class RNG:
    _rng = None

    @classmethod
    def get_rng(cls):
        if cls._rng is None:
            # seed = Config.get_manual_seed()
            # if seed is None:
            #     raise RuntimeError("Manual seed is not set in the configuration.")
            cls._rng = np.random.default_rng(0)
        return cls._rng

