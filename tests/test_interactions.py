import numpy as np

import pytest

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.interactions import Interactions


def _test_just_padding(sequences):
    """
    There should be no rows with only padding in them.
    """

    row_sum = sequences.sum(axis=1)

    assert len(row_sum) == sequences.shape[0]
    assert np.all(row_sum > 0)


def _test_final_column_no_padding(sequences):
    """
    The final column should always have an interaction.
    """

    assert np.all(sequences[:, -1] > 0)


def _test_shifted(sequence_users, sequences, step_size):
    """
    Unless there was a change of user, row i + 1's interactions
    should contain row i's interactions shifted to the right by
    step size.
    """

    for i in range(1, len(sequences)):

        if sequence_users[i] != sequence_users[i - 1]:
            # Change of user
            continue

        assert np.all(sequences[i][step_size:] == sequences[i - 1][:-step_size])


def _test_temporal_order(sequence_users, sequences, interactions):

    interaction_matrix = interactions.tocoo()
    interaction_matrix.data = interactions.timestamps
    interaction_matrix = interaction_matrix.tocsr().todense()

    for i, sequence in enumerate(sequences):

        user_id = sequence_users[i]
        nonpadded_sequence = sequence[sequence != 0]

        for j in range(0, len(nonpadded_sequence) - 1):
            item_id = nonpadded_sequence[j]

            next_item_id = nonpadded_sequence[j + 1]

            item_timestamp = interaction_matrix[user_id, item_id]
            next_item_timestamp = interaction_matrix[user_id, next_item_id]

            assert item_timestamp <= next_item_timestamp


def test_known_output_step_1():

    interactions = Interactions(np.zeros(5),
                                np.arange(5) + 1,
                                timestamps=np.arange(5))
    sequences = interactions.to_sequence(max_sequence_length=5,
                                         step_size=1).sequences

    expected = np.array([
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4],
        [0, 0, 1, 2, 3],
        [0, 0, 0, 1, 2],
        [0, 0, 0, 0, 1]
    ])

    assert np.all(sequences == expected)


def test_known_output_step_2():

    interactions = Interactions(np.zeros(5),
                                np.arange(5) + 1,
                                timestamps=np.arange(5))
    sequences = interactions.to_sequence(max_sequence_length=5,
                                         step_size=2).sequences

    expected = np.array([
        [1, 2, 3, 4, 5],
        [0, 0, 1, 2, 3],
        [0, 0, 0, 0, 1],
    ])

    assert np.all(sequences == expected)


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

    sequences = interactions.to_sequence(
        max_sequence_length=max_sequence_length,
        step_size=step_size)

    if step_size == 1:
        assert sequences.sequences.shape == (len(interactions),
                                             max_sequence_length)
    else:
        assert sequences.sequences.shape[1] == max_sequence_length

    _test_just_padding(sequences.sequences)
    _test_final_column_no_padding(sequences.sequences)
    _test_shifted(sequences.user_ids,
                  sequences.sequences,
                  step_size)
    _test_temporal_order(sequences.user_ids,
                         sequences.sequences,
                         interactions)


def test_to_sequence_min_length():

    min_sequence_length = 10
    interactions = movielens.get_movielens_dataset('100K')

    # Check that with default arguments there are sequences
    # that are shorter than we want
    sequences = interactions.to_sequence(max_sequence_length=20)
    assert np.any((sequences.sequences != 0).sum(axis=1) < min_sequence_length)

    # But no such sequences after we specify min length.
    sequences = interactions.to_sequence(max_sequence_length=20,
                                         min_sequence_length=min_sequence_length)
    assert not np.any((sequences.sequences != 0).sum(axis=1) < min_sequence_length)
