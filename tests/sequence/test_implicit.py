import numpy as np

import pytest

from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets import movielens, synthetic
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel


RANDOM_STATE = np.random.RandomState(42)


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.10),
    (1e2, 0.005),
])
def test_implicit_pooling_synthetic(randomness, expected_mrr):

    interactions = synthetic.generate_sequential(num_users=1000,
                                                 num_items=1000,
                                                 num_interactions=10000,
                                                 concentration_parameter=randomness,
                                                 random_state=RANDOM_STATE)
    train, test = user_based_train_test_split(interactions,
                                              random_state=RANDOM_STATE)

    train = train.to_sequence(max_sequence_length=30)
    test = test.to_sequence(max_sequence_length=30)

    model = ImplicitSequenceModel(loss='bpr',
                                  batch_size=1024,
                                  learning_rate=1e-1,
                                  l2=1e-9,
                                  n_iter=3)
    model.fit(train, verbose=True)
    mrr = sequence_mrr_score(model, test)

    print('MRR {} randomness {}'.format(mrr.mean(), randomness))

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.10),
    (1e2, 0.005),
])
def test_implicit_lstm_synthetic(randomness, expected_mrr):

    interactions = synthetic.generate_sequential(num_users=1000,
                                                 num_items=1000,
                                                 num_interactions=10000,
                                                 concentration_parameter=randomness,
                                                 random_state=RANDOM_STATE)
    train, test = user_based_train_test_split(interactions,
                                              random_state=RANDOM_STATE)

    train = train.to_sequence(max_sequence_length=10)
    test = test.to_sequence(max_sequence_length=10)

    model = ImplicitSequenceModel(loss='bpr',
                                  representation='lstm',
                                  batch_size=128,
                                  learning_rate=1e-1,
                                  l2=1e-7,
                                  n_iter=300)
    model.fit(train, verbose=True)
    mrr = sequence_mrr_score(model, test)

    print('MRR {} randomness {}'.format(mrr.mean(), randomness))

    assert mrr.mean() > expected_mrr


@pytest.mark.parametrize('randomness, expected_mrr', [
    (1e-3, 0.10),
    (1e2, 0.005),
])
def test_implicit_cnn_synthetic(randomness, expected_mrr):

    interactions = synthetic.generate_sequential(num_users=1000,
                                                 num_items=1000,
                                                 num_interactions=10000,
                                                 concentration_parameter=randomness,
                                                 random_state=RANDOM_STATE)
    train, test = user_based_train_test_split(interactions,
                                              random_state=RANDOM_STATE)

    train = train.to_sequence(max_sequence_length=10)
    test = test.to_sequence(max_sequence_length=10)

    model = ImplicitSequenceModel(loss='bpr',
                                  representation='cnn',
                                  batch_size=128,
                                  learning_rate=1e-2,
                                  l2=1e-7,
                                  n_iter=10,
                                  random_state=RANDOM_STATE)
    model.fit(train, verbose=True)
    mrr = sequence_mrr_score(model, test)

    print('MRR {} randomness {}'.format(mrr.mean(), randomness))

    assert mrr.mean() > expected_mrr


# def test_implicit_pooling_model():

#     interactions = movielens.get_movielens_dataset('100K')



#     train, test = user_based_train_test_split(interactions,
#                                               random_state=RANDOM_STATE)

#     train = train.to_sequence(max_sequence_length=30, min_sequence_length=30)
#     test = test.to_sequence(max_sequence_length=30, min_sequence_length=30)

#     model = ImplicitPoolingModel(loss='bpr',
#                                  embedding_dim=128,
#                                  batch_size=1024,
#                                  learning_rate=1e-1,
#                                  l2=1e-9,
#                                  n_iter=3)
#     model.fit(test, verbose=True)

#     mrr = sequence_mrr_score(model, test)

#     print('MRR {}'.format(mrr.mean()))
#     print('Full {}'.format(mrr[(test.sequences > 0).sum(1) > 5].mean()))
