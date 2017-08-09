import numpy as np

from spotlight.datasets import amazon, movielens, synthetic


def test_dataset_downloading():

    for variant in movielens.VARIANTS[:2]:
        movielens.get_movielens_dataset(variant)


def test_generate_content_based():

    interactions = synthetic.generate_content_based(num_users=10,
                                                    num_items=15,
                                                    num_user_features=3,
                                                    num_interactions=100)

    assert len(interactions) == 100
    assert np.all(interactions.user_ids < 10)
    assert np.all(interactions.item_ids < 15)


def test_amazon_dataset():

    amazon.get_amazon()
