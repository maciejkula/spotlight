import numpy as np

import torch

from spotlight.datasets.amazon import get_amazon
from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import mrr_score


def fit(train, test, validation, random_state):

    model = ImplicitFactorizationModel(batch_size=4096,
                                       use_cuda=torch.cuda.is_available())

    print('Training')
    model.fit(validation, verbose=True)

    print('Validating')
    test_mrr = mrr_score(model, test)
    val_mrr = mrr_score(model, validation)

    print('Test MRR {}, validation MRR {}'.format(
        test_mrr, val_mrr))


if __name__ == '__main__':

    random_state = np.random.RandomState(100)
    dataset = get_amazon()

    dataset.item_features = None

    train, rest = random_train_test_split(dataset,
                                          random_state=random_state)
    test, validation = random_train_test_split(rest,
                                               random_state=random_state)

    fit(train, test, validation, random_state)
