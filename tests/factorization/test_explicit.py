import os

import numpy as np
import pytest

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.layers import BloomEmbedding


RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))
# Acceptable variation in specific test runs
EPSILON = .005


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

    assert rmse - EPSILON < 1.0


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

    assert rmse - EPSILON < 1.0


def test_logistic():

    interactions = movielens.get_movielens_dataset('100K')

    # Convert to binary
    interactions.ratings = (interactions.ratings > 3).astype(np.float32)
    # Convert from (0, 1) to (-1, 1)
    interactions.ratings = interactions.ratings * 2 - 1

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ExplicitFactorizationModel(loss='logistic',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-3,
                                       l2=1e-6,
                                       use_cuda=CUDA)
    model.fit(train)

    rmse = rmse_score(model, test)

    assert rmse - EPSILON < 1.05


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


@pytest.mark.parametrize('compression_ratio, expected_rmse', [
    (0.2, 1.5),
    (0.5, 1.5),
    (1.0, 1.5),
    (1.5, 1.5),
    (2.0, 1.5),
])
def test_bloom(compression_ratio, expected_rmse):

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    user_embeddings = BloomEmbedding(interactions.num_users, 32,
                                     compression_ratio=compression_ratio,
                                     num_hash_functions=2)
    item_embeddings = BloomEmbedding(interactions.num_items, 32,
                                     compression_ratio=compression_ratio,
                                     num_hash_functions=2)
    network = BilinearNet(interactions.num_users,
                          interactions.num_items,
                          user_embedding_layer=user_embeddings,
                          item_embedding_layer=item_embeddings)

    model = ExplicitFactorizationModel(loss='regression',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-5,
                                       representation=network,
                                       use_cuda=CUDA)

    model.fit(train)
    print(model)

    rmse = rmse_score(model, test)
    print(rmse)

    assert rmse - EPSILON < expected_rmse
