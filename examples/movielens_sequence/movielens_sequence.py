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

NUM_SAMPLES = 100

LEARNING_RATES = [1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'hinge', 'adaptive_hinge', 'pointwise']
BATCH_SIZE = [8, 16, 32, 256]
EMBEDDING_DIM = [8, 16, 32, 64, 128, 256]
N_ITER = list(range(5, 20))
L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]


class Results:

    def __init__(self, filename):

        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, test_mrr, validation_mrr):

        result = {'test_mrr': test_mrr,
                  'validation_mrr': validation_mrr,
                  'hash': self._hash(hyperparams)}
        result.update(hyperparams)

        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):

        results = sorted([x for x in self],
                         key=lambda x: -x['test_mrr'])

        if results:
            return results[0]
        else:
            return None

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
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
        'kernel_width': [3, 5, 7],
        'num_layers': list(range(1, 10)),
        'dilation_multiplier': [1, 2],
        'nonlinearity': ['tanh', 'relu'],
        'residual': [True, False]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        params['dilation'] = list(params['dilation_multiplier'] ** (i % 8)
                                  for i in range(params['num_layers']))

        yield params


def sample_lstm_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:

        yield params


def sample_pooling_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:

        yield params


def evaluate_cnn_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    net = CNNNet(train.num_items,
                 embedding_dim=h['embedding_dim'],
                 kernel_width=h['kernel_width'],
                 dilation=h['dilation'],
                 num_layers=h['num_layers'],
                 nonlinearity=h['nonlinearity'],
                 residual_connections=h['residual'])

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
    val_mrr = sequence_mrr_score(model, validation)

    return test_mrr, val_mrr


def evaluate_lstm_model(hyperparameters, train, test, validation, random_state):

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
    val_mrr = sequence_mrr_score(model, validation)

    return test_mrr, val_mrr


def evaluate_pooling_model(hyperparameters, train, test, validation, random_state):

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
    val_mrr = sequence_mrr_score(model, validation)

    return test_mrr, val_mrr


def run(train, test, validation, ranomd_state, model_type):

    results = Results('{}_results.txt'.format(model_type))

    best_result = results.best()

    if model_type == 'pooling':
        eval_fnc, sample_fnc = (evaluate_pooling_model,
                                sample_pooling_hyperparameters)
    elif model_type == 'cnn':
        eval_fnc, sample_fnc = (evaluate_cnn_model,
                                sample_cnn_hyperparameters)
    elif model_type == 'lstm':
        eval_fnc, sample_fnc = (evaluate_lstm_model,
                                sample_lstm_hyperparameters)
    else:
        raise ValueError('Unknown model type')

    if best_result is not None:
        print('Best {} result: {}'.format(model_type, results.best()))

    for hyperparameters in sample_fnc(random_state, NUM_SAMPLES):

        if hyperparameters in results:
            continue

        print('Evaluating {}'.format(hyperparameters))

        (test_mrr, val_mrr) = eval_fnc(hyperparameters,
                                       train,
                                       test,
                                       validation,
                                       random_state)

        print('Test MRR {} val MRR {}'.format(
            test_mrr.mean(), val_mrr.mean()
        ))

        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean())

    return results


if __name__ == '__main__':

    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200
    random_state = np.random.RandomState(100)

    dataset = get_movielens_dataset('1M')

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)
    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)
    validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                        min_sequence_length=min_sequence_length,
                                        step_size=step_size)

    mode = sys.argv[1]

    run(train, test, validation, random_state, mode)
