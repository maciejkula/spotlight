import numpy as np

import pytest

from spotlight.datasets import movielens


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


def _test_shifted(sequences, targets):
    """
    Unless there was a change of user, row i + 1's interactions
    should contain row i's interactions shifted to the right by
    one.
    """

    for i in range(1, len(sequences)):

        if sequences[i - 1][:-1].sum() == 0:
            # Change of user, all columns but one
            # are padding.
            continue

        # The last element of the previous sequence
        # is this sequence's target.
        assert sequences[i - 1][-1] == targets[i]
        assert np.all(sequences[i][1:] == sequences[i - 1][:-1])


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


@pytest.mark.parametrize("max_sequence_length", [
    5,
    20,
    1024,
])
def test_to_sequence(max_sequence_length):

    interactions = movielens.get_movielens_dataset('100K')

    sequences = interactions.to_sequence(
        max_sequence_length=max_sequence_length)

    assert sequences.sequences.shape == (len(interactions) -
                                         len(np.unique(interactions.user_ids)),
                                         min(max_sequence_length,
                                             (interactions
                                              .tocsr()
                                              .getnnz(axis=1).max())))

    _test_just_padding(sequences.sequences)
    _test_final_column_no_padding(sequences.sequences)
    _test_shifted(sequences.sequences,
                  sequences.targets)
    _test_temporal_order(sequences.user_ids,
                         sequences.sequences,
                         interactions)
