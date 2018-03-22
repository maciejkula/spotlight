import os

import numpy as np

import pytest

from spotlight.evaluation import precision_recall_score, intra_distance_score
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.factorization.implicit import ImplicitFactorizationModel

RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


@pytest.fixture(scope='module')
def data():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=1,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       random_state=RANDOM_STATE,
                                       use_cuda=CUDA)
    model.fit(train)

    return train, test, model


@pytest.mark.parametrize('k', [
    1,
    [1, 1],
    [1, 1, 1]
])
def test_precision_recall(data, k):

    (train, test, model) = data

    precision, recall = precision_recall_score(model, test, train, k=k)

    assert precision.shape == recall.shape

    if not isinstance(k, list):
        assert len(precision.shape) == 1
    else:
        assert precision.shape[1] == len(k)


def test_intra_distance(data):

    (train, test, model) = data

    k = 5
    distances = intra_distance_score(model, test, train, k=k)

    assert len(distances) == test.num_users
    assert len(distances[0]) == (k * (k - 1)) / 2
