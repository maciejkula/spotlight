import os

import numpy as np

import torch

import pytest

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens, synthetic
from spotlight.evaluation import mrr_score


from spotlight.factorization.implicit import ImplicitFactorizationModel


RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


def _min_max_scale(arr):

    arr_min = arr.min()
    arr_max = arr.max()

    return (arr - arr_min) / (arr_max - arr_min)


@pytest.mark.parametrize('weight_factor, loss, expected', [
    (1, 'pointwise', 0.05),
    (0, 'pointwise', 0.05),
    (None, 'pointwise', 0.05),
    (1, 'bpr', 0.07),
    (0, 'bpr', 0.07),
    (None, 'bpr', 0.07),
    (1, 'hinge', 0.07),
    (0, 'hinge', 0.07),
    (None, 'hinge', 0.07),
    (1, 'adaptive_hinge', 0.07),
    (0, 'adaptive_hinge', 0.07),
    (None, 'adaptive_hinge', 0.07)
])
def test_weights(weight_factor, loss, expected):

    interactions = movielens.get_movielens_dataset('100K')

    # Add weights
    if weight_factor is not None:
        interactions.weights = np.repeat(weight_factor, len(interactions))

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss=loss,
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    if weight_factor == 0:
        assert mrr < expected
    else:
        assert mrr > expected


def test_bpr_custom_optimizer():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    def adagrad_optimizer(model_params,
                          lr=1e-2,
                          weight_decay=1e-6):

        return torch.optim.Adagrad(model_params,
                                   lr=lr,
                                   weight_decay=weight_decay)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       optimizer_func=adagrad_optimizer,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr > 0.05


@pytest.mark.parametrize('use_timestamps, expected_mrr', [
    (True, 0.06),
    (False, 0.060),
])
def test_bpr_hybrid(use_timestamps, expected_mrr):

    interactions = movielens.get_movielens_dataset('100K')

    if use_timestamps:
        normalized_timestamps = (_min_max_scale(interactions.timestamps)
                                 .reshape(-1, 1))
        interactions.context_features = normalized_timestamps / 100

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6)
    model.fit(train, verbose=True)
    print(model)

    mrr = mrr_score(model, test, train=train, average_per_context=False).mean()

    print('MRR {}'.format(mrr))

    assert mrr > expected_mrr


@pytest.mark.parametrize('use_features, expected_mrr', [
    (False, 0.01),
    (True, 0.11),
])
def test_synthetic_hybrid(use_features, expected_mrr):

    interactions = synthetic.generate_content_based(random_state=RANDOM_STATE)

    if not use_features:
        interactions.user_features = None
        interactions.context_features = None
        interactions.item_features = None

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=0.0,
                                       random_state=RANDOM_STATE)

    model.fit(train, verbose=True)
    print(model._net)

    mrr = mrr_score(model, test, train=train, average_per_context=False).mean()

    print('MRR {}'.format(mrr))

    assert mrr > expected_mrr
