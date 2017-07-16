import numpy as np

from spotlight import cross_validation
from spotlight.datasets import movielens


RANDOM_STATE = np.random.RandomState(42)


def test_user_based_split():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = (cross_validation
                   .user_based_train_test_split(interactions,
                                                test_percentage=0.2,
                                                random_state=RANDOM_STATE))

    assert len(train) + len(test) == len(interactions)

    users_in_test = len(np.unique(test.user_ids))
    assert np.allclose(float(users_in_test) / interactions.num_users,
                       0.2, atol=0.001)
