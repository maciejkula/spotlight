import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np

from sklearn.model_selection import ParameterSampler

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.datasets.amazon import get_amazon_dataset
from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import LSTMNet
from spotlight.layers import BloomEmbedding
from spotlight.evaluation import sequence_mrr_score


CUDA = (os.environ.get('CUDA') is not None or
        shutil.which('nvidia-smi') is not None)

NUM_SAMPLES = 50

LEARNING_RATES = [1e-4, 5 * 1e-4, 1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'adaptive_hinge']
BATCH_SIZE = [16, 32, 256]
EMBEDDING_DIM = [8, 16, 32, 64, 128, 256]
N_ITER = list(range(5, 20))
L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]


class Results:

    def __init__(self, filename):

        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, test_mrr, validation_mrr, elapsed):

        result = hyperparams.copy()

        result.update({'test_mrr': test_mrr,
                       'validation_mrr': validation_mrr,
                       'elapsed': elapsed,
                       'hash': self._hash(hyperparams)})

        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best_baseline(self):

        results = sorted([x for x in self if x['compression_ratio'] == 1.0],
                         key=lambda x: -x['test_mrr'])

        if results:
            return results[0]
        else:
            return None

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


def sample_hyperparameters(random_state, num):

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


def evaluate_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    if h['compression_ratio'] < 1.0:
        item_embeddings = BloomEmbedding(train.num_items, h['embedding_dim'],
                                         compression_ratio=h['compression_ratio'],
                                         num_hash_functions=4)
        network = LSTMNet(train.num_items, h['embedding_dim'],
                          item_embedding_layer=item_embeddings)
    else:
        network = 'lstm'

    model = ImplicitSequenceModel(loss=h['loss'],
                                  n_iter=h['n_iter'],
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  representation=network,
                                  use_cuda=CUDA,
                                  random_state=random_state)

    start_time = time.time()
    model.fit(train, verbose=True)
    elapsed = time.time() - start_time
    print(model)

    test_mrr = sequence_mrr_score(model, test)
    val_mrr = sequence_mrr_score(model, validation)

    return test_mrr, val_mrr, elapsed


def run(dataset, train, test, validation, random_state):

    results = Results('{}_results.txt'.format(dataset))
    compression_ratios = (np.arange(1, 10) / 10).tolist()

    best_result = results.best()

    if best_result is not None:
        print('Best result: {}'.format(results.best()))

    # Find a good baseline
    for hyperparameters in sample_hyperparameters(random_state, NUM_SAMPLES):

        print('Fitting {}'.format(hyperparameters))

        hyperparameters['compression_ratio'] = 1.0
        (test_mrr, val_mrr, elapsed) = evaluate_model(hyperparameters,
                                                      train,
                                                      test,
                                                      validation,
                                                      random_state)
        print('Test MRR {} val MRR {}'.format(
            test_mrr.mean(), val_mrr.mean()
        ))

        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean(), elapsed)

    best_baseline = results.best_baseline()
    print('Best baseline: {}'.format(best_baseline))

    # Compute compression results
    for compression_ratio in compression_ratios:

        hyperparameters = best_baseline
        hyperparameters['compression_ratio'] = compression_ratio

        if hyperparameters in results:
            continue

        print('Evaluating {}'.format(hyperparameters))

        (test_mrr, val_mrr, elapsed) = evaluate_model(hyperparameters,
                                                      train,
                                                      test,
                                                      validation,
                                                      random_state)
        print('Test MRR {} val MRR {}'.format(
            test_mrr.mean(), val_mrr.mean()
        ))

        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean(), elapsed)

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)

    args = parser.parse_args()

    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200

    random_state = np.random.RandomState(100)

    if args.dataset == 'movielens':
        dataset = get_movielens_dataset('1M')
    else:
        dataset = get_amazon_dataset()

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

    run(args.dataset, train, test, validation, random_state)
