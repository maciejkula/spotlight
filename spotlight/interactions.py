"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np

import scipy.sparse as sp

import torch

from torch.autograd import Variable

from spotlight.helpers import iter_none, make_tuple
from spotlight.torch_utils import gpu, grouped_minibatch, minibatch


def _sliding_window(tensor, window_size, step_size=1):

    for i in range(len(tensor), 0, -step_size):
        yield tensor[max(i - window_size, 0):i]


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length,
                        step_size):

    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length,
                                   step_size):

            yield (user_ids[i], seq)


def _slice_or_none(arg, slc):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return tuple(x[slc] for x in arg)
    else:
        return arg[slc]


def _tensor_or_none(arg, use_cuda):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return tuple(gpu(torch.from_numpy(x), use_cuda) for x in arg)
    else:
        return gpu(torch.from_numpy(arg), use_cuda)


def _variable_or_none(arg):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return tuple(Variable(x) for x in arg)
    else:
        return Variable(arg)


def _dim_or_zero(arg, axis=1):

    if arg is None:
        return 0
    elif isinstance(arg, tuple):
        return sum(x.shape[axis] for x in arg)
    else:
        return arg.shape[axis]


def _float_or_none(arg):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return (x.astype(np.float32) for x in arg)
    else:
        return arg.astype(np.float32)


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

    user_ids: array of np.int64
        array of user ids of the user-item pairs
    item_ids: array of np.int64
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
                 user_features=None,
                 item_features=None,
                 context_features=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids.astype(np.int64)
        self.item_ids = item_ids.astype(np.int64)
        self.ratings = _float_or_none(ratings)
        self.timestamps = timestamps
        self.weights = _float_or_none(weights)

        self.user_features = _float_or_none(user_features)
        self.item_features = _float_or_none(item_features)
        self.context_features = _float_or_none(context_features)

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

        for feature in make_tuple(self.user_features):
            if feature.shape[0] != self.num_users:
                raise ValueError('Number of user features not '
                                 'equal to number of users.')

        for feature in make_tuple(self.item_features):
            if feature.shape[0] != self.num_items:
                raise ValueError('Number of item features not '
                                 'equal to number of items.')

        for feature in make_tuple(self.context_features):
            if feature.shape[0] != len(self):
                raise ValueError('Number of context features not '
                                 'equal to number of interactions.')

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

    def _sort(self, indices):

        self.user_ids = self.user_ids[indices]
        self.item_ids = self.item_ids[indices]
        self.ratings = _slice_or_none(self.ratings, indices)
        self.timestamps = _slice_or_none(self.timestamps, indices)
        self.weights = _slice_or_none(self.timestamps, indices)
        self.context_features = _slice_or_none(self.context_features, indices)

    def shuffle(self, random_state=None):

        if random_state is None:
            random_state = np.random.RandomState()

        shuffle_indices = np.arange(len(self.user_ids))
        random_state.shuffle(shuffle_indices)

        self._sort(shuffle_indices)

    def minibatches(self, use_cuda=False, batch_size=128):

        batch_generator = zip(*(minibatch(*make_tuple(_tensor_or_none(attr, use_cuda)),
                                          batch_size=batch_size)
                                if attr is not None
                                else iter_none()
                                for attr in (self.user_ids,
                                             self.item_ids,
                                             self.ratings,
                                             self.timestamps,
                                             self.weights,
                                             self.context_features)))

        user_features = _tensor_or_none(self.user_features, use_cuda)
        item_features = _tensor_or_none(self.item_features, use_cuda)

        for (uids_batch, iids_batch, ratings_batch, timestamps_batch,
             weights_batch, cf_batch) in batch_generator:

            yield InteractionsMinibatch(
                user_ids=uids_batch,
                item_ids=iids_batch,
                ratings=ratings_batch,
                timestamps=timestamps_batch,
                weights=weights_batch,
                user_features=_slice_or_none(user_features, uids_batch),
                item_features=item_features,
                context_features=cf_batch
            )

    def contexts(self, use_cuda=False):

        if self.num_context_features():
            for batch in self.minibatches(use_cuda=use_cuda, batch_size=1):
                yield batch
        else:
            # Sort by user id
            sort_indices = np.argsort(self.user_ids)
            self._sort(sort_indices)

            batch_generator = zip(*(grouped_minibatch(
                self.user_ids,
                *make_tuple(_tensor_or_none(attr, use_cuda)))
                                    if attr is not None
                                    else iter_none()
                                    for attr in (self.user_ids,
                                                 self.item_ids,
                                                 self.ratings,
                                                 self.timestamps,
                                                 self.weights,
                                                 self.context_features)))

            user_features = _tensor_or_none(self.user_features, use_cuda)
            item_features = _tensor_or_none(self.item_features, use_cuda)

            for (uids_batch, iids_batch, ratings_batch, timestamps_batch,
                 weights_batch, cf_batch) in batch_generator:

                yield InteractionsMinibatch(
                    user_ids=uids_batch,
                    item_ids=iids_batch,
                    ratings=ratings_batch,
                    timestamps=timestamps_batch,
                    weights=weights_batch,
                    user_features=_slice_or_none(user_features, uids_batch),
                    item_features=item_features,
                    context_features=cf_batch
                )

    def num_user_features(self):

        return _dim_or_zero(self.user_features)

    def num_context_features(self):

        return _dim_or_zero(self.context_features)

    def num_item_features(self):

        return _dim_or_zero(self.item_features)

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

    def to_sequence(self, max_sequence_length=10, min_sequence_length=None, step_size=None):
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

        num_subsequences = int(np.ceil(counts / float(step_size)).sum())

        sequences = np.zeros((num_subsequences, max_sequence_length),
                             dtype=np.int32)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int32)
        for i, (uid,
                seq) in enumerate(_generate_sequences(user_ids,
                                                      item_ids,
                                                      indices,
                                                      max_sequence_length,
                                                      step_size)):
            sequences[i][-len(seq):] = seq
            sequence_users[i] = uid

        if min_sequence_length is not None:
            long_enough = sequences[:, -min_sequence_length] != 0
            sequences = sequences[long_enough]
            sequence_users = sequence_users[long_enough]

        return (SequenceInteractions(sequences,
                                     user_ids=sequence_users,
                                     num_items=self.num_items))


class InteractionsMinibatch(object):

    def __init__(self,
                 user_ids,
                 item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 user_features=None,
                 item_features=None,
                 context_features=None):

        self.user_ids = _variable_or_none(user_ids)
        self.item_ids = _variable_or_none(item_ids)
        self.ratings = _variable_or_none(ratings)
        self.timestamps = _variable_or_none(timestamps)
        self.weights = _variable_or_none(weights)

        self.user_features = _variable_or_none(user_features)
        self.item_features = item_features
        self.context_features = _variable_or_none(context_features)

    def get_item_features(self, item_ids):

        if isinstance(item_ids, Variable):
            item_ids = item_ids.data

        return _variable_or_none(_slice_or_none(self.item_features, item_ids))

    def __len__(self):

        return len(self.user_ids)


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
                 user_ids=None, num_items=None):

        self.sequences = sequences
        self.user_ids = user_ids
        self.max_sequence_length = sequences.shape[1]

        if num_items is None:
            self.num_items = sequences.max() + 1
        else:
            self.num_items = num_items
