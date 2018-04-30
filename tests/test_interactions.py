import numpy as np

import pytest

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.interactions import Interactions, _generate_sequences


def _test_shifted(sequences, step_size):
    """
    Unless there was a change of user, row i + 1's interactions
    should contain row i's interactions shifted to the right by
    step size.
    """

    previous_uid = None
    previous_sequence = None

    for user_id, sequence in sequences:
        if previous_uid == user_id:
            assert (np.all(sequence[-len(previous_sequence) + step_size:] ==
                           previous_sequence[:-step_size]))

        previous_uid = user_id
        previous_sequence = sequence


def _test_temporal_order(sequences, interactions):

    interaction_matrix = interactions.tocoo()
    interaction_matrix.data = interactions.timestamps
    interaction_matrix = interaction_matrix.tocsr().todense()

    for user_id, sequence in sequences:
        for j in range(0, len(sequence) - 1):
            item_id = sequence[j]

            next_item_id = sequence[j + 1]

            item_timestamp = interaction_matrix[user_id, item_id]
            next_item_timestamp = interaction_matrix[user_id, next_item_id]

            assert item_timestamp <= next_item_timestamp


def test_known_output_step_1():

    interactions = Interactions(np.zeros(5),
                                np.arange(5) + 1,
                                timestamps=np.arange(5))
    sequences = list(v.tolist() for (_, v) in _generate_sequences(interactions,
                                                                  max_sequence_length=5,
                                                                  step_size=1))

    expected = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4],
        [1, 2, 3],
        [1, 2],
        [1]
    ]

    assert sequences == expected


def test_known_output_step_2():

    interactions = Interactions(np.zeros(5),
                                np.arange(5) + 1,
                                timestamps=np.arange(5))
    sequences = list(v.tolist() for (_, v) in _generate_sequences(interactions,
                                                                  max_sequence_length=5,
                                                                  step_size=2))

    expected = [
        [1, 2, 3, 4, 5],
        [1, 2, 3],
        [1],
    ]

    assert sequences == expected


@pytest.mark.parametrize('max_sequence_length, step_size', [
    (5, 1),
    (5, 3),
    (20, 1),
    (20, 4),
    (1024, 1024),
    (1024, 5)
])
def test_to_sequence(max_sequence_length, step_size):

    interactions = movielens.get_movielens_dataset('100K')
    _, interactions = random_train_test_split(interactions)

    def seqs():
        return _generate_sequences(
            interactions,
            max_sequence_length=max_sequence_length,
            step_size=step_size)

    _test_shifted(seqs(),
                  step_size)
    _test_temporal_order(seqs(),
                         interactions)


def test_to_sequence_min_length():

    min_sequence_length = 10
    interactions = movielens.get_movielens_dataset('100K')

    def seqs(min_sequence_length):
        return _generate_sequences(
            interactions,
            max_sequence_length=10,
            min_sequence_length=min_sequence_length,
            step_size=1)

    # Check that with default arguments there are sequences
    # that are shorter than we want
    assert any(len(v) < min_sequence_length for (_, v) in seqs(min_sequence_length=1))

    # But no such sequences after we specify min length.
    assert not any(len(v) < min_sequence_length for (_, v) in seqs(min_sequence_length=20))
