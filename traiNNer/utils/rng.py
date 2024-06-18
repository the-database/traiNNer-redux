import numpy as np

from utils.config import Config

seed = Config.get_manual_seed()


rng = np.random.default_rng(seed)
print(f"init rng with seed = {seed}")
