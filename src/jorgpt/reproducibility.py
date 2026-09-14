import random

import numpy as np
import torch


def set_seed(seed: int = 12) -> None:
    """Fija las semillas aleatorias principales de JorGPT."""

    if seed < 0:
        raise ValueError("seed debe ser mayor o igual que 0.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
