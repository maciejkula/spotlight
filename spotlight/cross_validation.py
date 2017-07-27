"""
Module with functionality for splitting and shuffling datasets.
"""

import numpy as np

from sklearn.utils import murmurhash3_32

from spotlight.interactions import Interactions, _slice_or_none


def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    interactions.shuffle(random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(interactions.user_ids[train_idx],
                         interactions.item_ids[train_idx],
                         ratings=_slice_or_none(interactions.ratings,
                                                train_idx),
                         timestamps=_slice_or_none(interactions.timestamps,
                                                   train_idx),
                         weights=_slice_or_none(interactions.weights,
                                                train_idx),
                         user_features=interactions.user_features,
                         context_features=_slice_or_none(interactions.context_features,
                                                         train_idx),
                         item_features=interactions.item_features,
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_slice_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_slice_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_slice_or_none(interactions.weights,
                                               test_idx),
                        user_features=interactions.user_features,
                        context_features=_slice_or_none(interactions.context_features,
                                                        test_idx),
                        item_features=interactions.item_features,
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test


def user_based_train_test_split(interactions,
                                test_percentage=0.2,
                                random_state=None):
    """
    Split interactions between a train and a test set based on
    user ids, so that a given user's entire interaction history
    is either in the train, or the test set.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of users to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    if random_state is None:
        random_state = np.random.RandomState()

    minint = np.iinfo(np.uint32).min
    maxint = np.iinfo(np.uint32).max

    seed = random_state.randint(minint, maxint)

    in_test = ((murmurhash3_32(interactions.user_ids.astype(np.int32),
                               seed=seed,
                               positive=True) % 100 /
                100.0) <
               test_percentage)
    in_train = np.logical_not(in_test)

    train = Interactions(interactions.user_ids[in_train],
                         interactions.item_ids[in_train],
                         ratings=_slice_or_none(interactions.ratings,
                                                in_train),
                         timestamps=_slice_or_none(interactions.timestamps,
                                                   in_train),
                         weights=_slice_or_none(interactions.weights,
                                                in_train),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[in_test],
                        interactions.item_ids[in_test],
                        ratings=_slice_or_none(interactions.ratings,
                                               in_test),
                        timestamps=_slice_or_none(interactions.timestamps,
                                                  in_test),
                        weights=_slice_or_none(interactions.weights,
                                               in_test),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test
