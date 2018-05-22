import os

import numpy as np

import pytest

from spotlight.evaluation import precision_recall_score, sequence_precision_recall_score
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence import ImplicitSequenceModel


RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


@pytest.fixture(scope='module')
def data_implicit_factorization():

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


@pytest.fixture(scope='module')
def data_implicit_sequence():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitSequenceModel(loss='adaptive_hinge',
                                  representation='lstm',
                                  batch_size=[8, 16, 32, 256],
                                  learning_rate=[1e-3, 1e-2, 5 * 1e-2, 1e-1],
                                  l2=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0],
                                  n_iter=list(range(1, 2)),
                                  use_cuda=CUDA,
                                  random_state=RANDOM_STATE)

    model.fit(train, verbose=True)

    return train, test, model


@pytest.mark.parametrize('k', 10)
def test_sequence_precision_recall(data_implicit_sequence, k):

    (train, test, model) = data_implicit_sequence

    interactions = movielens.get_movielens_dataset('100K')
    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    precision, recall = sequence_precision_recall_score(model, test, train, k=k)

    assert precision.shape == recall.shape

    if not isinstance(k, list):
        assert len(precision.shape) == 1
    else:
        assert precision.shape[1] == len(k)


@pytest.mark.parametrize('k', [
    1,
    [1, 1],
    [1, 1, 1]
])
def test_precision_recall(data_implicit_factorization, k):

    (train, test, model) = data_implicit_factorization

    interactions = movielens.get_movielens_dataset('100K')
    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    precision, recall = precision_recall_score(model, test, train, k=k)

    assert precision.shape == recall.shape

    if not isinstance(k, list):
        assert len(precision.shape) == 1
    else:
        assert precision.shape[1] == len(k)
