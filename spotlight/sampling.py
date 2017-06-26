import numpy as np

import torch

from torch.autograd import Variable

from spotlight.torch_utils import gpu


def sample_items(num_items, num_samples, random_state=None, use_cuda=False):

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, num_samples)

    return Variable(gpu(torch.from_numpy(items)))
