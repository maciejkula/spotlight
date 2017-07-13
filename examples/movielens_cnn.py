import hashlib
import json
import os
import shutil
import sys

import numpy as np

from sklearn.model_selection import ParameterSampler

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score


CUDA = (os.environ.get('CUDA') is not None or
        shutil.which('nvidia-smi') is not None)


class Results:

    def __init__(self, filename):

        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, mrr):

        result = {'mrr': mrr, 'hash': self._hash(hyperparams)}
        result.update(hyperparams)

        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def __getitem__(self, hyperparams):

        params_hash = self._hash(hyperparams)

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum

        raise KeyError

    def __contains__(self, x):

        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                del datum['hash']

                yield datum


def sample_cnn_hyperparameters(random_state, num):

    space = {
        'n_iter': list(range(5, 30)),
        'batch_size': [256, 512, 1024],
        'l2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0],
        'learning_rate': [1e-3, 1e-2, 5 * 1e-2],
        'loss': ['bpr', 'hinge', 'pointwise'],
        'num_layers': list(range(1, 10)),
        'embedding_dim': [8, 16, 32, 64, 128, 256],
        'dilation_multiplier': [1, 2, 3],
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        params['dilation'] = list(params['dilation_multiplier'] ** i
                                  for i in range(params['num_layers']))

        yield params


def sample_lstm_hyperparameters(random_state, num):

    space = {
        'n_iter': list(range(5, 30)),
        'batch_size': [256, 512, 1024],
        'l2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0],
        'learning_rate': [1e-3, 1e-2, 5 * 1e-2],
        'loss': ['bpr', 'hinge', 'pointwise'],
        'embedding_dim': [8, 16, 32, 64, 128, 256]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:

        yield params


def sample_pooling_hyperparameters(random_state, num):

    space = {
        'n_iter': list(range(5, 30)),
        'batch_size': [256, 512, 1024],
        'l2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0],
        'learning_rate': [1e-3, 1e-2, 5 * 1e-2],
        'loss': ['bpr', 'hinge', 'pointwise'],
        'embedding_dim': [8, 16, 32, 64, 128, 256]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:

        yield params


def evaluate_cnn_model(hyperparameters, train, test, random_state):

    h = hyperparameters

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

    return test_mrr


def evaluate_lstm_model(hyperparameters, train, test, random_state):

    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='lstm',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)

    return test_mrr


def evaluate_pooling_model(hyperparameters, train, test, random_state):

    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='pooling',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)

    return test_mrr


def run_cnn(train, test, random_state):

    results = Results('cnn_results.txt')

    for hyperparameters in sample_cnn_hyperparameters(random_state, 1000):

        if hyperparameters in results:
            print('Already computed, skipping...')
            continue

        print('Evaluating {}'.format(hyperparameters))

        mrr = evaluate_cnn_model(hyperparameters,
                                 train,
                                 test,
                                 random_state)

        print('Test MRR {}'.format(
            mrr.mean()
        ))

        results.save(hyperparameters, mrr.mean())

    return results


def run_pooling(train, test, random_state):

    results = Results('pooling_results.txt')

    for hyperparameters in sample_pooling_hyperparameters(random_state, 1000):

        if hyperparameters in results:
            print('Already computed, skipping...')
            continue

        print('Evaluating {}'.format(hyperparameters))

        mrr = evaluate_pooling_model(hyperparameters,
                                     train,
                                     test,
                                     random_state)

        print('Test MRR {}'.format(
            mrr.mean()
        ))

        results.save(hyperparameters, mrr.mean())

    return results


def run_lstm(train, test, random_state):

    results = Results('lstm_results.txt')

    for hyperparameters in sample_pooling_hyperparameters(random_state, 1000):

        if hyperparameters in results:
            print('Already computed, skipping...')
            continue

        print('Evaluating {}'.format(hyperparameters))

        mrr = evaluate_lstm_model(hyperparameters,
                                  train,
                                  test,
                                  random_state)

        print('Test MRR {}'.format(
            mrr.mean()
        ))

        results.save(hyperparameters, mrr.mean())

    return results


if __name__ == '__main__':

    max_sequence_length = 100
    min_sequence_length = 20
    random_state = np.random.RandomState(100)

    dataset = get_movielens_dataset('100K')

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)
    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length)
    validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                        min_sequence_length=min_sequence_length)

    mode = sys.argv[1]

    if mode == 'cnn':
        run_cnn(train, test, random_state)
    elif mode == 'pooling':
        run_pooling(train, test, random_state)
    elif mode == 'lstm':
        run_lstm(train, test, random_state)
    else:
        raise ValueError('Unknown model type')
