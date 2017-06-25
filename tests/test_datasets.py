from spotlight.datasets import movielens, _transport


def test_dataset_downloading():

    for variant in movielens.VARIANTS[:2]:
        movielens.get_movielens_dataset(variant)
