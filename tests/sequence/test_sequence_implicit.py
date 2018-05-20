import os

import numpy as np

import pytest

from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets import synthetic
from spotlight.evaluation import sequence_mrr_score
from spotlight.layers import BloomEmbedding
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet, LSTMNet, PoolNet


RANDOM_SEED = 42
NUM_EPOCHS = 5
EMBEDDING_DIM = 32
BATCH_SIZE = 128
LOSS = 'bpr'
VERBOSE = True
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


def _get_synthetic_data(num_users=100,
                        num_items=100,
                        num_interactions=10000,
                        randomness=0.01,
                        order=2,
                        max_sequence_length=10,
                        random_state=None):

    interactions = synthetic.generate_sequential(num_users=num_users,
                                                 num_items=num_items,
                                                 num_interactions=num_interactions,
                                                 concentration_parameter=randomness,
                                                 order=order,
                                                 random_state=random_state)

    print('Max prob {}'.format((np.unique(interactions.item_ids,
                                          return_counts=True)[1] /
                                num_interactions).max()))

    train, test = user_based_train_test_split(interactions,
                                              random_state=random_state)

    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              step_size=None)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            step_size=None)

    return train, test


def _evaluate(model, test):

    test_mrr = sequence_mrr_score(model, test)

    print('Test MRR {}'.format(
        test_mrr.mean()
    ))

    return test_mrr


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.18),
    (1e2, 0.03),
])
def test_implicit_pooling_synthetic(randomness, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=randomness,
                                      random_state=random_state)

    model = ImplicitSequenceModel(loss=LOSS,
                                  batch_size=BATCH_SIZE,
                                  embedding_dim=EMBEDDING_DIM,
                                  learning_rate=1e-1,
                                  l2=1e-9,
                                  n_iter=NUM_EPOCHS,
                                  random_state=random_state,
                                  use_cuda=CUDA)
    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.61),
    (1e2, 0.03),
])
def test_implicit_lstm_synthetic(randomness, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=randomness,
                                      random_state=random_state)

    model = ImplicitSequenceModel(loss=LOSS,
                                  representation='lstm',
                                  batch_size=BATCH_SIZE,
                                  embedding_dim=EMBEDDING_DIM,
                                  learning_rate=1e-2,
                                  l2=1e-7,
                                  n_iter=NUM_EPOCHS * 5,
                                  random_state=random_state,
                                  use_cuda=CUDA)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.65),
    (1e2, 0.03),
])
def test_implicit_cnn_synthetic(randomness, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=randomness,
                                      random_state=random_state)

    model = ImplicitSequenceModel(loss=LOSS,
                                  representation=CNNNet(train.num_items,
                                                        embedding_dim=EMBEDDING_DIM,
                                                        kernel_width=5,
                                                        num_layers=1),
                                  batch_size=BATCH_SIZE,
                                  learning_rate=1e-2,
                                  l2=0.0,
                                  n_iter=NUM_EPOCHS * 5,
                                  random_state=random_state,
                                  use_cuda=CUDA)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('num_layers, dilation, expected_mrr', [
    (1, (1,), 0.65),
    (2, (1, 2), 0.65),
])
def test_implicit_cnn_dilation_synthetic(num_layers, dilation, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=1e-03,
                                      num_interactions=20000,
                                      random_state=random_state)

    model = ImplicitSequenceModel(loss=LOSS,
                                  representation=CNNNet(train.num_items,
                                                        embedding_dim=EMBEDDING_DIM,
                                                        kernel_width=3,
                                                        dilation=dilation,
                                                        num_layers=num_layers),
                                  batch_size=BATCH_SIZE,
                                  learning_rate=1e-2,
                                  l2=0.0,
                                  n_iter=NUM_EPOCHS * 5 * num_layers,
                                  random_state=random_state,
                                  use_cuda=CUDA)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.3),
    (1e2, 0.03),
])
def test_implicit_lstm_mixture_synthetic(randomness, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=randomness,
                                      random_state=random_state)

    model = ImplicitSequenceModel(loss=LOSS,
                                  representation='mixture',
                                  batch_size=BATCH_SIZE,
                                  embedding_dim=EMBEDDING_DIM,
                                  learning_rate=1e-2,
                                  l2=1e-7,
                                  n_iter=NUM_EPOCHS * 10,
                                  random_state=random_state,
                                  use_cuda=CUDA)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('loss, expected_mrr', [
    ('pointwise', 0.15),
    ('hinge', 0.16),
    ('bpr', 0.18),
    ('adaptive_hinge', 0.16),
])
def test_implicit_pooling_losses(loss, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=1e-3,
                                      random_state=random_state)

    model = ImplicitSequenceModel(loss=loss,
                                  batch_size=BATCH_SIZE,
                                  embedding_dim=EMBEDDING_DIM,
                                  learning_rate=1e-1,
                                  l2=1e-9,
                                  n_iter=NUM_EPOCHS,
                                  random_state=random_state,
                                  use_cuda=CUDA)
    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('compression_ratio, expected_mrr', [
    (0.2, 0.14),
    (0.5, 0.30),
    (1.0, 0.5),
])
def test_bloom_cnn(compression_ratio, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=1e-03,
                                      num_interactions=20000,
                                      random_state=random_state)

    embedding = BloomEmbedding(train.num_items,
                               32,
                               compression_ratio=compression_ratio,
                               num_hash_functions=2)

    representation = CNNNet(train.num_items,
                            embedding_dim=EMBEDDING_DIM,
                            kernel_width=3,
                            item_embedding_layer=embedding)

    model = ImplicitSequenceModel(loss=LOSS,
                                  representation=representation,
                                  batch_size=BATCH_SIZE,
                                  learning_rate=1e-2,
                                  l2=0.0,
                                  n_iter=NUM_EPOCHS,
                                  random_state=random_state,
                                  use_cuda=CUDA)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('compression_ratio, expected_mrr', [
    (0.2, 0.18),
    (0.5, 0.40),
    (1.0, 0.60),
])
def test_bloom_lstm(compression_ratio, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=1e-03,
                                      num_interactions=20000,
                                      random_state=random_state)

    embedding = BloomEmbedding(train.num_items,
                               32,
                               compression_ratio=compression_ratio,
                               num_hash_functions=4)

    representation = LSTMNet(train.num_items,
                             embedding_dim=EMBEDDING_DIM,
                             item_embedding_layer=embedding)

    model = ImplicitSequenceModel(loss=LOSS,
                                  representation=representation,
                                  batch_size=BATCH_SIZE,
                                  learning_rate=1e-2,
                                  l2=1e-7,
                                  n_iter=NUM_EPOCHS * 5,
                                  random_state=random_state,
                                  use_cuda=CUDA)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('compression_ratio, expected_mrr', [
    (0.2, 0.06),
    (0.5, 0.07),
    (1.0, 0.13),
])
def test_bloom_pooling(compression_ratio, expected_mrr):

    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(randomness=1e-03,
                                      num_interactions=20000,
                                      random_state=random_state)

    embedding = BloomEmbedding(train.num_items,
                               32,
                               compression_ratio=compression_ratio,
                               num_hash_functions=2)

    representation = PoolNet(train.num_items,
                             embedding_dim=EMBEDDING_DIM,
                             item_embedding_layer=embedding)

    model = ImplicitSequenceModel(loss=LOSS,
                                  representation=representation,
                                  batch_size=BATCH_SIZE,
                                  learning_rate=1e-2,
                                  l2=1e-7,
                                  n_iter=NUM_EPOCHS * 5,
                                  random_state=random_state,
                                  use_cuda=CUDA)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr
