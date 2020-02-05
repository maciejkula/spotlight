import os

import numpy as np

import pytest

from spotlight.evaluation import precision_recall_score, sequence_precision_recall_score
from spotlight.cross_validation import random_train_test_split, user_based_train_test_split
from spotlight.datasets import movielens
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel

RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))
# Acceptable variation in specific test runs
EPSILON = .005


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

    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200

    interactions = movielens.get_movielens_dataset('100K')

    train, test = user_based_train_test_split(interactions,
                                              random_state=RANDOM_STATE)

    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)

    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)

    model = ImplicitSequenceModel(loss='adaptive_hinge',
                                  representation='lstm',
                                  batch_size=8,
                                  learning_rate=1e-2,
                                  l2=1e-3,
                                  n_iter=2,
                                  use_cuda=CUDA,
                                  random_state=RANDOM_STATE)

    model.fit(train, verbose=True)

    return train, test, model


@pytest.mark.parametrize('k', [10])
def test_sequence_precision_recall(data_implicit_sequence, k):

    (train, test, model) = data_implicit_sequence

    precision, recall = sequence_precision_recall_score(model, test, k)
    precision = precision.mean()
    recall = recall.mean()

    # with respect to the hyper-parameters specified in data_implicit_sequence
    expected_precision = 0.064
    expected_recall = 0.064

    # true_pos/(true_pos + false_pos) == true_pos/(true_pos + false_neg)
    # because num_predictions is set equal to num_targets in sequence_precision_recall_score
    assert precision == recall
    assert expected_precision - EPSILON < precision and precision < expected_precision + EPSILON
    assert expected_recall - EPSILON < recall and recall < expected_recall + EPSILON


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
