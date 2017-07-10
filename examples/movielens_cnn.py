import json
import os
import shutil

import numpy as np

from scipy.stats import distributions

from sklearn.model_selection import ParameterSampler

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score


CUDA = (os.environ.get('CUDA') is not None or
        shutil.which('nvidia-smi') is not None)


def sample_hyperparameters(random_state, num):

    space = {
        'n_iter': distributions.randint(5, 30),
        'batch_size': [256, 512, 1024],
        'l2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0],
        'learning_rate': [1e-3, 1e-2, 5 * 1e-2],
        'loss': ['bpr', 'hinge', 'pointwise'],
        'num_layers': distributions.randint(1, 10),
        'embedding_dim': [8, 16, 32, 64, 128, 256]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        params['dilation'] = list(2 ** i for i in range(params['num_layers']))

        yield params


def evaluate_model(hyperparameters, train, test, random_state):

    h = hyperparameters
    print('Evaluating {}'.format(hyperparameters))

    net = CNNNet(train.num_items,
                 embedding_dim=h['embedding_dim'],
                 kernel_width=3,
                 dilation=h['dilation'],
                 num_layers=h['num_layers'])

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation=net,
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)

    print('Test MRR {}'.format(
        test_mrr.mean()
    ))

    h['mrr'] = test_mrr.mean()

    return h


def is_saved(filename, hyperparams):

    with open(filename, 'r') as result_file:
        for line in result_file:

            line_result = json.loads(line)
            del line_result['mrr']

            if line_result == hyperparams:
                return True

    return False


if __name__ == '__main__':

    sequence_length = 50
    random_state = np.random.RandomState(100)

    dataset = get_movielens_dataset('100K')

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   random_state=random_state)
    train = train.to_sequence(sequence_length)
    test = test.to_sequence(sequence_length)
    validation = validation.to_sequence(sequence_length)

    with open('results.txt', 'a') as output:
        for hyperparams in sample_hyperparameters(random_state, 1000):

            if is_saved('results.txt', hyperparams):
                print('Already computed')
                continue

            result = evaluate_model(hyperparams,
                                    train,
                                    test,
                                    random_state)
            output.write(json.dumps(result) + '\n')
            output.flush()
