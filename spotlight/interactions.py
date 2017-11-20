"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np

import scipy.sparse as sp

import torch

from dynarray import DynamicArray

from spotlight.torch_utils import gpu


def _sliding_window(tensor, window_size, step_size=1):

    for i in range(len(tensor), 0, -step_size):
        yield tensor[max(i - window_size, 0):i]


def _generate_batched_sequences(item_ids,
                                indices,
                                max_sequence_length,
                                min_sequence_length,
                                step_size):

    packed = {}

    for i in range(len(indices)):
        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = len(item_ids)
        else:
            stop_idx = indices[i + 1]

        for subsequence in _sliding_window(item_ids[start_idx:stop_idx],
                                           max_sequence_length,
                                           step_size=step_size):

            if len(subsequence) < min_sequence_length:
                continue

            (packed.setdefault(len(subsequence),
                               DynamicArray((None, len(subsequence)), dtype=np.int64))
             .append(subsequence))

    for value in packed.values():
        value.shrink_to_fit()

    return {key: value[:] for (key, value) in packed.items()}


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.

    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.

    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

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
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
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
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
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

        if self.user_ids.max() >= self.num_users:
            raise ValueError('Maximum user id greater '
                             'than declared number of users.')
        if self.item_ids.max() >= self.num_items:
            raise ValueError('Maximum item id greater '
                             'than declared number of items.')

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

    def to_sequence(self, max_sequence_length=10, min_sequence_length=1, step_size=None):
        """
        Transform to sequence form.

        User-item interaction pairs are sorted by their timestamps,
        and sequences of up to max_sequence_length events are arranged
        into a (zero-padded from the left) matrix with dimensions
        (num_sequences x max_sequence_length).

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5], the
        returned interactions matrix at sequence length 5 and step size
        1 will be be given by:

        .. code-block:: python

           [[1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1]]

        At step size 2:

        .. code-block:: python

           [[1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 0, 1]]

        Parameters
        ----------

        max_sequence_length: int, optional
            Maximum sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        min_sequence_length: int, optional
            If set, only sequences with at least min_sequence_length
            non-padding elements will be returned.
        step-size: int, optional
            The returned subsequences are the effect of moving a
            a sliding window over the input. This parameter
            governs the stride of that window. Increasing it will
            result in fewer subsequences being returned.

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

        if step_size is None:
            step_size = max_sequence_length

        # Sort first by user id, then by timestamp
        sort_indices = np.lexsort((self.timestamps,
                                   self.user_ids))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        sequences = _generate_batched_sequences(item_ids,
                                                indices,
                                                max_sequence_length,
                                                min_sequence_length,
                                                step_size)

        return (SequenceInteractions(sequences,
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

    Attributes
    ----------

    sequences: array of np.int32 of shape (num_sequences x max_sequence_length)
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    """

    def __init__(self,
                 sequences,
                 num_items=None):

        self.sequences = sequences

        self._num_sequences = sum(x.shape[0] for x in self.sequences.values())
        self._max_sequence_length = max(x.shape[1] for x in self.sequences.values())

        if num_items is None:
            self.num_items = max(x.max() + 1 for x in self.sequences.values())
        else:
            self.num_items = num_items

    def __repr__(self):

        return ('<Sequence interactions dataset ({num_sequences} '
                'sequences x {sequence_length} max sequence length)>'
                .format(
                    num_sequences=self._num_sequences,
                    sequence_length=self._max_sequence_length,
                ))

    def __iter__(self):

        for value in self.sequences.values():
            for row in value:
                yield row

    def minibatch(self, batch_size, random_state=None, use_cuda=False, only_full_batches=True):
        """
        Iterate over minibatches of the dataset. Each minibatch has the same sequence length,
        but the lengths of each minibatche can differ to avoid padding. Minibatch order as
        well as sequences within a minibatch are shuffled on each epoch.

        Parameters
        ----------

        batch_size: int
            The size of each minibatch. Minibatches smaller than this will not be emitted
            if `only_full_batches` is `True` (default).
        random_state: instance of numpy.random.RandomState, optional
            Random generator to use when returning the minibatches.
        use_cida: bool, optional
            Whether to send the minibatch data to the GPU.
        only_full_batches: bool, optional
            If `True`, minibatches smaller than `batch_size` are omitted. This is helpful
            if loss values are averaged across minibatch. In that case, minibatches with
            few examples would have a disproportionately large effect on the gradients.
        """

        if random_state is None:
            random_state = np.random.RandomState()

        # Shuffle the sequences within blocks of same length.
        for value in self.sequences.values():
            random_state.shuffle(value)

        # Build a list of minibatches to execute that
        # randomly alternates between the lengths.
        minibatches = []

        for key, value in self.sequences.items():
            for i in range(0, len(value), batch_size):
                minibatches.append([key, i, i + batch_size])

        minibatches = np.array(minibatches, dtype=np.int64)
        random_state.shuffle(minibatches)

        # Convert to tensors (and possibly transfer to GPU).
        tensor_data = {k: gpu(torch.from_numpy(v)) for (k, v) in self.sequences.items()}

        for (key, start, stop) in minibatches:
            btch = tensor_data[key][start:stop]

            if len(btch) != batch_size:
                continue

            yield btch
