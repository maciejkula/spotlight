import numpy as np

import pytest

from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets import synthetic
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet


RANDOM_SEED = 42
NUM_EPOCHS = 5
EMBEDDING_DIM = 32
BATCH_SIZE = 128
LOSS = 'bpr'
VERBOSE = False


def _get_synthetic_data(num_users=100,
                        num_items=100,
                        num_interactions=10000,
                        randomness=0.01,
                        order=2,
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

    train = train.to_sequence(max_sequence_length=10)
    test = test.to_sequence(max_sequence_length=10)

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
                                  random_state=random_state)
    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.65),
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
                                  n_iter=NUM_EPOCHS,
                                  random_state=random_state)

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
                                  n_iter=NUM_EPOCHS,
                                  random_state=random_state)

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
                                  n_iter=NUM_EPOCHS * num_layers,
                                  random_state=random_state)

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('loss, expected_mrr', [
    ('pointwise', 0.16),
    ('hinge', 0.17),
    ('bpr', 0.19),
    ('adaptive_hinge', 0.17),
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
                                  random_state=random_state)
    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    assert mrr.mean() > expected_mrr
