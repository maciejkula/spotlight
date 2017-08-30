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
from spotlight.cross_validation import (random_train_test_split,
                                        user_based_train_test_split)
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.representations import LSTMNet
from spotlight.factorization.representations import BilinearNet
from spotlight.layers import BloomEmbedding, ScaledEmbedding
from spotlight.evaluation import mrr_score, sequence_mrr_score
from spotlight.torch_utils import set_seed


CUDA = (os.environ.get('CUDA') is not None or
        shutil.which('nvidia-smi') is not None)

NUM_SAMPLES = 50

LEARNING_RATES = [1e-4, 5 * 1e-4, 1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'adaptive_hinge']
BATCH_SIZE = [16, 32, 64, 128, 256, 512]
EMBEDDING_DIM = [32, 64, 128, 256]
N_ITER = list(range(1, 20))
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

        results = sorted([x for x in self
                          if x['compression_ratio'] == 1.0 and
                          x['embedding_dim'] >= 32],
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


def build_factorization_model(hyperparameters, train, random_state):
    h = hyperparameters

    set_seed(42, CUDA)

    if h['compression_ratio'] < 1.0:
        item_embeddings = BloomEmbedding(train.num_items, h['embedding_dim'],
                                         compression_ratio=h['compression_ratio'],
                                         num_hash_functions=4,
                                         padding_idx=0)
        user_embeddings = BloomEmbedding(train.num_users, h['embedding_dim'],
                                         compression_ratio=h['compression_ratio'],
                                         num_hash_functions=4,
                                         padding_idx=0)
    else:
        item_embeddings = ScaledEmbedding(train.num_items, h['embedding_dim'],
                                          padding_idx=0)
        user_embeddings = ScaledEmbedding(train.num_users, h['embedding_dim'],
                                          padding_idx=0)

    network = BilinearNet(train.num_users,
                          train.num_items,
                          user_embedding_layer=user_embeddings,
                          item_embedding_layer=item_embeddings)

    model = ImplicitFactorizationModel(loss=h['loss'],
                                       n_iter=h['n_iter'],
                                       batch_size=h['batch_size'],
                                       learning_rate=h['learning_rate'],
                                       embedding_dim=h['embedding_dim'],
                                       l2=h['l2'],
                                       representation=network,
                                       use_cuda=CUDA,
                                       random_state=np.random.RandomState(42))

    return model


def build_sequence_model(hyperparameters, train, random_state):

    h = hyperparameters

    set_seed(42, CUDA)

    if h['compression_ratio'] < 1.0:
        item_embeddings = BloomEmbedding(train.num_items, h['embedding_dim'],
                                         compression_ratio=h['compression_ratio'],
                                         num_hash_functions=4,
                                         padding_idx=0)
    else:
        item_embeddings = ScaledEmbedding(train.num_items, h['embedding_dim'],
                                          padding_idx=0)

    network = LSTMNet(train.num_items, h['embedding_dim'],
                      item_embedding_layer=item_embeddings)

    model = ImplicitSequenceModel(loss=h['loss'],
                                  n_iter=h['n_iter'],
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  embedding_dim=h['embedding_dim'],
                                  l2=h['l2'],
                                  representation=network,
                                  use_cuda=CUDA,
                                  random_state=np.random.RandomState(42))

    return model


def evaluate_model(model, train, test, validation):

    start_time = time.time()
    model.fit(train, verbose=True)
    elapsed = time.time() - start_time

    print('Elapsed {}'.format(elapsed))
    print(model)

    if hasattr(test, 'sequences'):
        test_mrr = sequence_mrr_score(model, test)
        val_mrr = sequence_mrr_score(model, validation)
    else:
        test_mrr = mrr_score(model, test)
        val_mrr = mrr_score(model, test.tocsr() + validation.tocsr())

    return test_mrr, val_mrr, elapsed


def run(experiment_name, train, test, validation, random_state):

    results = Results('{}_results.txt'.format(experiment_name))
    compression_ratios = (np.arange(1, 10) / 10).tolist()

    best_result = results.best()

    if best_result is not None:
        print('Best result: {}'.format(results.best()))

    # Find a good baseline
    for hyperparameters in sample_hyperparameters(random_state, NUM_SAMPLES):

        if 'factorization' in experiment_name:
            # We want bigger batches for factorization models
            hyperparameters['batch_size'] = hyperparameters['batch_size'] * 4

        hyperparameters['compression_ratio'] = 1.0

        if hyperparameters in results:
            print('Done, skipping...')
            continue

        if 'factorization' in experiment_name:
            model = build_factorization_model(hyperparameters,
                                              train,
                                              random_state)
        else:
            model = build_sequence_model(hyperparameters,
                                         train,
                                         random_state)

        print('Fitting {}'.format(hyperparameters))
        (test_mrr, val_mrr, elapsed) = evaluate_model(model,
                                                      train,
                                                      test,
                                                      validation)

        print('Test MRR {} val MRR {} elapsed {}'.format(
            test_mrr.mean(), val_mrr.mean(), elapsed
        ))

        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean(), elapsed)

    best_baseline = results.best_baseline()
    print('Best baseline: {}'.format(best_baseline))

    # Compute compression results
    for compression_ratio in compression_ratios:

        hyperparameters = best_baseline
        hyperparameters['compression_ratio'] = compression_ratio

        if hyperparameters in results:
            print('Compression computed')
            continue

        if 'factorization' in experiment_name:
            model = build_factorization_model(hyperparameters,
                                              train,
                                              random_state)
        else:
            model = build_sequence_model(hyperparameters,
                                         train,
                                         random_state)

        print('Evaluating {}'.format(hyperparameters))

        (test_mrr, val_mrr, elapsed) = evaluate_model(model,
                                                      train,
                                                      test,
                                                      validation)
        print('Test MRR {} val MRR {} elapsed {}'.format(
            test_mrr.mean(), val_mrr.mean(), elapsed
        ))

        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean(), elapsed)

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('model', type=str)

    args = parser.parse_args()

    random_state = np.random.RandomState(100)

    if args.dataset == 'movielens':
        dataset = get_movielens_dataset('1M')
        test_percentage = 0.2
    else:
        test_percentage = 0.01
        dataset = get_amazon_dataset(min_user_interactions=20,
                                     min_item_interactions=5)

    print(dataset)

    if args.model == 'sequence':
        max_sequence_length = int(np.percentile(dataset.tocsr()
                                                .getnnz(axis=1),
                                                95))
        min_sequence_length = 20
        step_size = max_sequence_length

        train, rest = user_based_train_test_split(dataset,
                                                  test_percentage=0.05,
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
        print('In test {}, in validation {}'.format(
            len(test.sequences),
            len(validation.sequences))
        )
    elif args.model == 'factorization':
        train, rest = random_train_test_split(dataset,
                                              test_percentage=test_percentage,
                                              random_state=random_state)
        test, validation = random_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)

    experiment_name = '{}_{}'.format(args.dataset, args.model)

    run(experiment_name, train, test, validation, random_state)
