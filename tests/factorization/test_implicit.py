import os

import numpy as np
import torch

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.evaluation import mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel


RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


def test_pointwise():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='pointwise',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr > 0.05


def test_bpr():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr > 0.07


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


def test_hinge():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='hinge',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr > 0.07


def test_adaptive_hinge():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='adaptive_hinge',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr = mrr_score(model, test, train=train).mean()

    assert mrr > 0.07
