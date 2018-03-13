import numpy as np
import torch
import pytest
from spotlight.torch_utils import minibatch


@pytest.mark.parametrize('n', [1, 12])
def test_minibatch(n):
    data = torch.randn(n, 6)
    s = 0
    ss = 0
    for x in minibatch(data, batch_size=3):
        s += np.prod(x.size())
        ss += x.sum()
        assert x.size(1) == data.size(1)
    assert s == np.prod(data.size())
    assert ss == data.sum()
