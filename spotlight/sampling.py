"""
Module containing functions for negative item sampling.
"""

import numpy as np


def sample_items(num_items, shape, random_state=None):
    """
    Randomly sample a number of items.

    Parameters
    ----------

    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.

    Returns
    -------

    items: np.array of shape [shape]
        Sampled item ids.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, shape, dtype=np.int64)

    return items
