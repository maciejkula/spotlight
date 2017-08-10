import hashlib
import json
import os
import shutil
import time

import numpy as np

from sklearn.model_selection import ParameterSampler

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.layers import BloomEmbedding
from spotlight.evaluation import mrr_score


CUDA = (os.environ.get('CUDA') is not None or
        shutil.which('nvidia-smi') is not None)

NUM_SAMPLES = 100

LEARNING_RATES = [1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'hinge', 'adaptive_hinge', 'pointwise']
BATCH_SIZE = [256, 512, 1024, 2048]
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

        result = {'test_mrr': test_mrr,
                  'validation_mrr': validation_mrr,
                  'elapsed': elapsed,
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


def sample_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
        'num_hash_functions': [1, 2, 3, 4, 5, 6]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        yield params


def evaluate_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    if h['compression_ratio'] < 1.0:
        user_embeddings = BloomEmbedding(train.num_users, h['embedding_dim'],
                                         compression_ratio=h['compression_ratio'],
                                         num_hash_functions=h['num_hash_functions'])
        item_embeddings = BloomEmbedding(train.num_items, h['embedding_dim'],
                                         compression_ratio=h['compression_ratio'],
                                         num_hash_functions=h['num_hash_functions'])
        network = BilinearNet(train.num_users,
                              train.num_items,
                              user_embedding_layer=user_embeddings,
                              item_embedding_layer=item_embeddings)
    else:
        network = None

    model = ImplicitFactorizationModel(loss=h['loss'],
                                       n_iter=h['n_iter'],
                                       batch_size=h['batch_size'],
                                       learning_rate=h['learning_rate'],
                                       l2=h['l2'],
                                       module=network,
                                       use_cuda=CUDA,
                                       random_state=random_state)

    start_time = time.time()
    model.fit(train, verbose=True)
    elapsed = time.time() - start_time

    test_mrr = mrr_score(model, test)
    val_mrr = mrr_score(model, validation)

    return test_mrr, val_mrr, elapsed


def run(train, test, validation, random_state):

    results = Results('results.txt')
    compression_ratios = (0.2, 0.5, 0.7, 1.0)

    best_result = results.best()

    if best_result is not None:
        print('Best result: {}'.format(results.best()))

    for hyperparameters in sample_hyperparameters(random_state, NUM_SAMPLES):
        for compression_ratio in compression_ratios:

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

    random_state = np.random.RandomState(100)

    dataset = get_movielens_dataset('1M')

    train, rest = random_train_test_split(dataset,
                                          random_state=random_state)
    test, validation = random_train_test_split(rest,
                                               test_percentage=0.5,
                                               random_state=random_state)

    run(train, test, validation, random_state)
