import numpy as np

import scipy.sparse as sp

from spotlight.torch_utils import minibatch


def _generate_sequences(item_ids, indices, max_sequence_length):

    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in minibatch(item_ids[start_idx:stop_idx],
                             batch_size=max_sequence_length):
            yield seq


class Interactions(object):

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

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):

        return self.tocoo().tocsr()

    def to_sequence(self, max_sequence_length=10):

        # Sort
        if self.timestamps is not None:
            # Sort first by user id, then by timestamp
            sort_indices = np.lexsort((self.timestamps,
                                       self.user_ids))
        else:
            sort_indices = np.argsort(self.user_ids)

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        _, indices = np.unique(user_ids, return_index=True)

        num_subsequences = sum(1 for _ in _generate_sequences(item_ids,
                                                              indices,
                                                              max_sequence_length))

        matrix = np.zeros((num_subsequences, max_sequence_length),
                          dtype=np.int32)

        for i, seq in enumerate(_generate_sequences(item_ids,
                                                    indices,
                                                    max_sequence_length)):
            matrix[i][-len(seq):] = seq

        return matrix
