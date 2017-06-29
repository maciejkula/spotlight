from spotlight.datasets import movielens


def test_to_sequence():

    interactions = movielens.get_movielens_dataset('100K')

    sequences = interactions.to_sequence()

    assert False
