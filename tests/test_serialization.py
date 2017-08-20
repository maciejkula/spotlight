import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.evaluation import mrr_score, sequence_mrr_score
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet


RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


def _reload(model):
    dirname = tempfile.mkdtemp()

    try:
        fname = os.path.join(dirname, "model.pkl")

        torch.save(model, fname)
        model = torch.load(fname)

    finally:
        shutil.rmtree(dirname)

    return model


@pytest.fixture(scope="module")
def data():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    return train, test


def test_explicit_serialization(data):

    train, test = data

    model = ExplicitFactorizationModel(loss='regression',
                                       n_iter=3,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-5,
                                       use_cuda=CUDA)
    model.fit(train)

    rmse_original = rmse_score(model, test)
    rmse_recovered = rmse_score(_reload(model), test)

    assert rmse_original == rmse_recovered


def test_implicit_serialization(data):

    train, test = data

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=3,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    mrr_original = mrr_score(model, test, train=train).mean()
    mrr_recovered = mrr_score(_reload(model), test, train=train).mean()

    assert mrr_original == mrr_recovered


def test_implicit_sequence_serialization(data):

    train, test = data
    train = train.to_sequence(max_sequence_length=128)
    test = test.to_sequence(max_sequence_length=128)

    model = ImplicitSequenceModel(loss='bpr',
                                  representation=CNNNet(train.num_items,
                                                        embedding_dim=32,
                                                        kernel_width=3,
                                                        dilation=(1, ),
                                                        num_layers=1),
                                  batch_size=128,
                                  learning_rate=1e-1,
                                  l2=0.0,
                                  n_iter=5,
                                  random_state=RANDOM_STATE,
                                  use_cuda=CUDA)
    model.fit(train)

    mrr_original = sequence_mrr_score(model, test).mean()
    mrr_recovered = sequence_mrr_score(_reload(model), test).mean()

    assert mrr_original == mrr_recovered
