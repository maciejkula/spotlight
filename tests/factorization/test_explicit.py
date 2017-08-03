import os

import numpy as np
import pytest

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel


RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


def test_regression():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ExplicitFactorizationModel(loss='regression',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-5,
                                       use_cuda=CUDA)
    model.fit(train)

    rmse = rmse_score(model, test)

    assert rmse < 1.0


def test_poisson():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ExplicitFactorizationModel(loss='poisson',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    rmse = rmse_score(model, test)

    assert rmse < 1.0


def test_check_input():
    # Train for single iter.
    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ExplicitFactorizationModel(loss='regression',
                                       n_iter=1,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    # Modify data to make imcompatible with original model.
    train.user_ids[0] = train.user_ids.max() + 1
    with pytest.raises(ValueError):
        model.fit(train)
