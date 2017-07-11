"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np

import scipy.sparse as sp


def _sliding_window(tensor, window_size):

    for i in range(len(tensor), 0, -1):
        yield tensor[max(i - window_size, 0):i]


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length):

    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length):

            yield (user_ids[i], seq)


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.

    Parameters
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self._check()

    def __repr__(self):

        return ('<Interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions)>'
                .format(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_interactions=len(self)
                ))

    def __len__(self):

        return len(self.user_ids)

    def _check(self):

        num_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids),
                            ('ratings', self.ratings),
                            ('timestamps', self.timestamps),
                            ('weights', self.weights)):

            if value is None:
                continue

            if len(value) != num_interactions:
                raise ValueError('Invalid {} dimensions: length '
                                 'must be equal to number of interactions'
                                 .format(name))

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, max_sequence_length=10, min_sequence_length=None):
        """
        Transform to sequence form.

        User-item interaction pairs are sorted by their timestamps,
        and sequences of up to max_sequence_length events are arranged
        into a (zero-padded from the left) matrix with dimensions
        (num_sequences x max_sequence_length).

        All valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [5, 2, 6, 3], the
        returned interactions matrix at sequence length 5 be given by:

        .. code-block:: python

           [[0, 5, 2, 6, 3],
            [0, 0, 5, 2, 6],
            [0, 0, 0, 5, 2],
            [0, 0, 0, 0, 5]]

        Parameters
        ----------

        max_sequence_length: int, optional
            maximum sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        min_sequence_length: int, optional
            If set, only sequences with at least min_sequence_length
            non-padding elements will be returned.

        Returns
        -------

        sequence interactions: :class:`~SequenceInteractions`
            The resulting sequence interactions.
        """

        if self.timestamps is None:
            raise ValueError('Cannot convert to sequences, '
                             'timestamps not available.')

        if 0 in self.item_ids:
            raise ValueError('0 is used as an item id, conflicting '
                             'with the sequence padding value.')

        # Sort first by user id, then by timestamp
        sort_indices = np.lexsort((self.timestamps,
                                   self.user_ids))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices = np.unique(user_ids, return_index=True)

        num_subsequences = len(self)

        sequences = np.zeros((num_subsequences, max_sequence_length),
                             dtype=np.int32)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int32)
        for i, (uid,
                seq) in enumerate(_generate_sequences(user_ids,
                                                      item_ids,
                                                      indices,
                                                      max_sequence_length)):
            sequences[i][-len(seq):] = seq
            sequence_users[i] = uid

        if min_sequence_length is not None:
            long_enough = sequences[:, -min_sequence_length] != 0
            sequences = sequences[long_enough]
            sequence_users = sequence_users[long_enough]

        return (SequenceInteractions(sequences,
                                     user_ids=sequence_users,
                                     num_items=self.num_items))


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------

    sequences: array of np.int32 of shape (num_sequences x max_sequence_length)
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    num_items: int, optional
        The number of distinct items in the data
    """

    def __init__(self,
                 sequences,
                 user_ids=None, num_items=None):

        self.sequences = sequences
        self.user_ids = user_ids
        self.max_sequence_length = sequences.shape[1]

        if num_items is None:
            self.num_items = sequences.max() + 1
        else:
            self.num_items = num_items
