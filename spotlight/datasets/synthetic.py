import time

import numpy as np

from spotlight.interactions import Interactions


def _build_transition_matrix(num_items,
                             concentration_parameter,
                             random_state):

    transition_matrix = np.cumsum(
        random_state.dirichlet(
            np.repeat(concentration_parameter, num_items),
            num_items),
        axis=1)

    return transition_matrix


def _generate_sequences(num_steps,
                        transition_matrix,
                        random_state):

    elements = []

    rvs = random_state.rand(num_steps)
    row_idx = random_state.randint(transition_matrix.shape[0])

    for rv in rvs:

        elements.append(row_idx + 1)

        row = transition_matrix[row_idx]
        row_idx = np.searchsorted(row, rv)

    return np.array(elements, dtype=np.int32)


def generate_sequential(num_users=100,
                        num_items=1000,
                        num_interactions=10000,
                        concentration_parameter=0.1,
                        random_state=None):

    if random_state is None:
        random_state = np.random.RandomState()

    transition_matrix = _build_transition_matrix(
        num_items - 1,
        concentration_parameter,
        random_state)

    user_ids = np.sort(random_state.randint(0,
                                            num_users,
                                            num_interactions,
                                            dtype=np.int32))
    item_ids = _generate_sequences(num_interactions,
                                   transition_matrix,
                                   random_state)
    timestamps = np.arange(len(user_ids), dtype=np.int32)
    ratings = np.ones(len(user_ids), dtype=np.float32)

    return Interactions(user_ids,
                        item_ids,
                        ratings=ratings,
                        timestamps=timestamps,
                        num_users=num_users,
                        num_items=num_items)
