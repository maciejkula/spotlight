import os
import pickle
import time

import numpy as np

import torch

from spotlight.layers import BloomEmbedding, ScaledEmbedding
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import LSTMNet

from spotlight.datasets.movielens import get_movielens_dataset


CUDA = torch.cuda.is_available()
EMBEDDING_DIM = 64
N_ITER = 2
NUM_HASH_FUNCTIONS = 4


def time_fitting(model, data, repetitions=2):

    timings = []

    # Warm-up epoch
    model.fit(data)

    for _ in range(repetitions):
        start_time = time.time()
        model.fit(data)
        timings.append(time.time() - start_time)

    print(min(timings))

    return min(timings)


def factorization_model(num_embeddings, bloom):

    if bloom:
        user_embeddings = BloomEmbedding(num_embeddings, EMBEDDING_DIM,
                                         num_hash_functions=NUM_HASH_FUNCTIONS)
        item_embeddings = BloomEmbedding(num_embeddings, EMBEDDING_DIM,
                                         num_hash_functions=NUM_HASH_FUNCTIONS)
    else:
        user_embeddings = ScaledEmbedding(num_embeddings, EMBEDDING_DIM)
        item_embeddings = ScaledEmbedding(num_embeddings, EMBEDDING_DIM)

    network = BilinearNet(num_embeddings,
                          num_embeddings,
                          user_embedding_layer=user_embeddings,
                          item_embedding_layer=item_embeddings)

    model = ImplicitFactorizationModel(loss='adaptive_hinge',
                                       n_iter=N_ITER,
                                       embedding_dim=EMBEDDING_DIM,
                                       batch_size=2048,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       representation=network,
                                       use_cuda=CUDA)

    return model


def sequence_model(num_embeddings, bloom):

    if bloom:
        item_embeddings = BloomEmbedding(num_embeddings, EMBEDDING_DIM,
                                         num_hash_functions=NUM_HASH_FUNCTIONS)
    else:
        item_embeddings = ScaledEmbedding(num_embeddings, EMBEDDING_DIM)

    network = LSTMNet(num_embeddings, EMBEDDING_DIM,
                      item_embedding_layer=item_embeddings)

    model = ImplicitSequenceModel(loss='adaptive_hinge',
                                  n_iter=N_ITER,
                                  batch_size=512,
                                  learning_rate=1e-3,
                                  l2=1e-2,
                                  representation=network,
                                  use_cuda=CUDA)

    return model


def get_sequence_data():

    dataset = get_movielens_dataset('1M')
    max_sequence_length = 200
    min_sequence_length = 20
    data = dataset.to_sequence(max_sequence_length=max_sequence_length,
                               min_sequence_length=min_sequence_length,
                               step_size=max_sequence_length)
    print(data.sequences.shape)

    return data


def get_factorization_data():

    dataset = get_movielens_dataset('1M')

    return dataset


def embedding_size_scalability():

    sequence_data = get_sequence_data()
    factorization_data = get_factorization_data()

    embedding_dims = (1e4,
                      1e4 * 5,
                      1e5,
                      1e5 * 5,
                      1e6,
                      1e6 * 5)

    bloom_sequence = np.array([time_fitting(sequence_model(int(dim), True),
                                            sequence_data)
                               for dim in embedding_dims])
    baseline_sequence = np.array([time_fitting(sequence_model(int(dim), False),
                                               sequence_data)
                                  for dim in embedding_dims])
    sequence_ratio = bloom_sequence / baseline_sequence

    print('Sequence ratio {}'.format(sequence_ratio))

    bloom_factorization = np.array([time_fitting(factorization_model(int(dim), True),
                                                 factorization_data)
                                    for dim in embedding_dims])
    baseline_factorization = np.array([time_fitting(factorization_model(int(dim), False),
                                                    factorization_data)
                                       for dim in embedding_dims])
    factorization_ratio = bloom_factorization / baseline_factorization

    print('Factorization ratio {}'.format(factorization_ratio))

    return np.array(embedding_dims), sequence_ratio, factorization_ratio


def plot(dims, sequence, factorization):

    import matplotlib
    matplotlib.use('Agg')  # NOQA
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("darkgrid")

    plt.ylabel("Speed improvement")
    plt.xlabel("Size of embedding layers")
    plt.title("Fitting speed (1.0 = no change)")
    plt.xscale('log')

    plt.plot(dims,
             1.0 / sequence,
             label='Sequence model')
    plt.plot(dims,
             1.0 / factorization,
             label='Factorization model')
    plt.legend(loc='lower right')
    plt.savefig('speed.png')
    plt.close()


if __name__ == '__main__':

    fname = 'performance.pickle'

    if not os.path.exists(fname):
        dims, sequence, factorization = embedding_size_scalability()
        with open(fname, 'wb') as fle:
            pickle.dump((dims, sequence, factorization), fle)

    with open(fname, 'rb') as fle:
        (dims, sequence, factorization) = pickle.load(fle)

    plot(dims, sequence, factorization)
