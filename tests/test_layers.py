import numpy as np
import pytest
import scipy.stats
import torch

import torch.nn as nn
from torch.autograd import Variable

from spotlight.layers import PRIMES, BloomEmbedding, ScaledEmbedding


class KnuthHash(object):
    """
    Roughly implements the hashing scheme used in
    :class:`spotlight.layers.BloomEmbedding` to
    test that it hashes relatively uniformly.
    """

    def __init__(self, m, k):

        self._m = m
        self._k = k

        self._masks = np.array(PRIMES[:self._k], dtype=np.int64)

    def hash(self, value):
        return (value * self._masks) % self._m


@pytest.mark.parametrize('m, k, num_observations, max_id', [
    (1000, 2, 10**3, 10**3),
    (1000, 6, 10**4, 10**3),
    (1000, 1, 10**4, 10**7),
])
def test_uniformity(m, k, num_observations, max_id):

    hasher = KnuthHash(m, k)

    indices = np.random.randint(0, max_id, size=num_observations)

    collisions = np.zeros(m)

    for idx in indices:
        collisions[hasher.hash(idx)] += 1

    collisions /= k

    _, p_value = scipy.stats.chisquare(collisions)

    assert p_value > 0.1


@pytest.mark.parametrize('embedding_class', [
    nn.Embedding,
    ScaledEmbedding,
    BloomEmbedding
])
def test_embeddings(embedding_class):

    num_embeddings = 1000
    embedding_dim = 16

    batch_size = 32
    sequence_length = 8

    layer = embedding_class(num_embeddings,
                            embedding_dim)

    # Test 1-d inputs (minibatch)
    indices = Variable(torch.from_numpy(
        np.random.randint(0, num_embeddings, size=batch_size, dtype=np.int64)))
    representation = layer(indices)
    assert representation.size() == (batch_size, embedding_dim)

    # Test 2-d inputs (minibatch x sequence_length)
    indices = Variable(torch.from_numpy(
        np.random.randint(0, num_embeddings,
                          size=(batch_size, sequence_length), dtype=np.int64)))
    representation = layer(indices)
    assert representation.size() == (batch_size, sequence_length, embedding_dim)
