import numpy as np

from spotlight.interactions import Interactions


def _index_or_none(array, shuffle_index):

    if array is None:
        return None
    else:
        return array[shuffle_index]


def shuffle_interactions(interactions,
                         random_state=None):

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(interactions.user_ids))
    random_state.shuffle(shuffle_indices)

    return Interactions(interactions.user_ids[shuffle_indices],
                        interactions.item_ids[shuffle_indices],
                        ratings=_index_or_none(interactions.ratings,
                                               shuffle_indices),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  shuffle_indices),
                        weights=_index_or_none(interactions.weights,
                                               shuffle_indices),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)


def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):

    interactions = shuffle_interactions(interactions,
                                        random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(interactions.user_ids[train_idx],
                         interactions.item_ids[train_idx],
                         ratings=_index_or_none(interactions.ratings,
                                                train_idx),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   train_idx),
                         weights=_index_or_none(interactions.weights,
                                                train_idx),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_index_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_index_or_none(interactions.weights,
                                               test_idx),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test
