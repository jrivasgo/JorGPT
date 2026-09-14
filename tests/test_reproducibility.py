import random

import numpy as np
import torch
import pytest

from jorgpt.reproducibility import set_seed


def test_reproducibility_with_same_seed():
    set_seed(12)

    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(3)

    set_seed(12)

    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(3)

    assert python_value_1 == python_value_2
    assert numpy_value_1 == numpy_value_2
    assert torch.equal(torch_value_1, torch_value_2)


def test_negative_seed_raises_error():
    with pytest.raises(ValueError):
        set_seed(-1)
