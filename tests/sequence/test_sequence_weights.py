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
                        random_state=None,
                        weight_type='ones'):

    interactions = synthetic.generate_sequential(num_users=num_users,
                                                 num_items=num_items,
                                                 num_interactions=num_interactions,
                                                 concentration_parameter=randomness,
                                                 order=order,
                                                 random_state=random_state,
                                                 weight_type=weight_type)

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


def _gen_model(max_sequence_length, weight_type):
    random_state = np.random.RandomState(RANDOM_SEED)
    train, test = _get_synthetic_data(random_state=random_state,
                                      max_sequence_length=max_sequence_length,
                                      weight_type=weight_type)

    model = ImplicitSequenceModel(
        loss=LOSS,
        representation=CNNNet(
            train.num_items,
            embedding_dim=EMBEDDING_DIM,
            kernel_width=5,
            num_layers=1
        ),
        batch_size=BATCH_SIZE,
        learning_rate=1e-2,
        l2=0.0,
        n_iter=NUM_EPOCHS * 5,
        random_state=random_state,
        use_cuda=CUDA
    )
    return model, train, test

@pytest.mark.parametrize('max_sequence_length', [(5),(10),(20)])
def test_zero_weights(max_sequence_length):

    model, train, test = _gen_model(max_sequence_length, 'zeros')

    try:
        model.fit(train, verbose=VERBOSE)
    except ValueError:
        # ValueError: Degenerate epoch loss: nan
        # in implicit.fit (due to 0-weights)
        pass

    mrr = _evaluate(model, test)

    random_mrr = 1.0/max_sequence_length

    assert mrr.mean() <= random_mrr


@pytest.mark.parametrize('max_sequence_length', [(10)])
def test_high_weights(max_sequence_length):

    model, train, test = _gen_model(max_sequence_length, 'high')

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    random_mrr = 1.0/max_sequence_length

    # high weights appear to have no overall effect on the model's divergence
    assert mrr.mean() >= random_mrr

@pytest.mark.parametrize('max_sequence_length', [(10)])
def test_ones_weights(max_sequence_length):

    model, train, test = _gen_model(max_sequence_length, 'ones')

    model.fit(train, verbose=VERBOSE)

    mrr = _evaluate(model, test)

    random_mrr = 1.0/max_sequence_length

    # high weights appear to have no overall effect on the model's divergence
    assert mrr.mean() >= random_mrr


# TODO def test_zero_user_weights(max_sequence_length):
# TODO an additional datasets.synthetic path for this


if __name__ == '__main__':
    test_ones_weights(10)
