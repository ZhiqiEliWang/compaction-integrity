import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int | None = None) -> int:
    if seed is None:
        seed = int(os.getenv("COMPACTION_SEED", str(DEFAULT_SEED)))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
