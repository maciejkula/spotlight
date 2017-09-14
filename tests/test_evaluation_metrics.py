import os

import numpy as np

from spotlight.evaluation import precision_recall_score
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.factorization.implicit import ImplicitFactorizationModel

RANDOM_STATE = np.random.RandomState(42)
CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', False))


def test_precision_recall():

    interactions = movielens.get_movielens_dataset('100K')

    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)

    model = ImplicitFactorizationModel(loss='bpr',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       random_state=RANDOM_STATE,
                                       use_cuda=CUDA)
    model.fit(train)

    precision, recall = precision_recall_score(model, test, train)

    print(np.mean(precision), np.mean(recall))
