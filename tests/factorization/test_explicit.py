import os

import numpy as np
import pytest

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.evaluation import rmse_score
from spotlight.losses import regression_loss, poisson_loss
from spotlight.factorization.explicit import ExplicitFactorizationModel

import torch
from torch.autograd import Variable

RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


def test_regression_loss():

    # Test loss function with zero weights
    observed_ratings = Variable(torch.from_numpy((2 * np.ones(100))))
    predicted_ratings = Variable(torch.from_numpy((np.ones(100))))
    weights = Variable(torch.from_numpy(np.zeros(100)))

    out = regression_loss(observed_ratings, predicted_ratings, weights)
    assert out.data.numpy().sum().item() == 0


def test_poisson_loss():

    # Test loss function with zero weights
    observed_ratings = Variable(torch.from_numpy((2 * np.ones(100))))
    predicted_ratings = Variable(torch.from_numpy((np.ones(100))))
    weights = Variable(torch.from_numpy(np.zeros(100)))

    out = poisson_loss(observed_ratings, predicted_ratings, weights)

    assert out.data.numpy().sum().item() == 0


@pytest.mark.parametrize('weight_factor, loss, expected', [
    (1, 'regression', 0.5),
    (0, 'regression', 1.0),
    (1, 'poisson', 1.0),
    (0, 'poisson', 2.0)
])
def test_model_fitting(weight_factor, loss, expected):

    interactions = movielens.get_movielens_dataset('100K')

    # Add weights
    interactions.weights = weight_factor * np.ones((100000,))

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ExplicitFactorizationModel(loss=loss,
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-5,
                                       use_cuda=CUDA)

    model.fit(train)

    rmse = rmse_score(model, test)

    assert rmse > expected


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
