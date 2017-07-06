import numpy as np


def sample_items(num_items, shape, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, shape)

    return items
