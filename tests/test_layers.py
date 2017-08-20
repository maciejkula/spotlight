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
    (100, 1, 10**4, 10**3),
    (100, 2, 10**5, 10**3),
    (100, 4, 10**6, 10**3),
])
def test_uniformity(m, k, num_observations, max_id):

    hasher = KnuthHash(m, k)

    indices = np.random.randint(0, max_id, size=num_observations)

    collisions = np.zeros(hasher._m)

    repeated_hashes = 0

    for idx in indices:
        hashes = hasher.hash(idx)
        repeated_hashes += len(hashes) - len(set(hashes))
        collisions[hashes] += 1

    assert repeated_hashes / k / num_observations < 0.2

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


def test_resize():
    """
    Test we can get three-dimensional indexing of embeddings layers via suitable
    resizing.
    """

    num_masks = 4
    layer = nn.Embedding(100, 10)
    original_shape = (20, 15, num_masks)
    resized_shape = (20, 60)

    indices_np = np.random.randint(0, 100, size=original_shape)

    indices = torch.from_numpy(indices_np)
    indices_resized = torch.from_numpy(indices_np).view(*resized_shape)

    embeddings_resized = layer(Variable(indices_resized))
    embeddings_resized = embeddings_resized.view(20, 15, 4, 10).mean(2)

    embeddings = layer(Variable(indices[:, :, 0]))
    for idx in range(1, num_masks):
        embeddings += layer(Variable(indices[:, :, idx]))
    embeddings /= num_masks

    assert (embeddings == embeddings_resized).data.numpy().all()
