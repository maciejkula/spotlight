import numpy as np
import pytest
import scipy.stats

from spotlight.layers import PRIMES


class KnuthHash(object):

    def __init__(self, m, k):

        self._m = m
        self._k = k

        self._masks = np.array(PRIMES[:self._k], dtype=np.int64)

    def hash(self, value):
        return (value * self._masks) % self._m


@pytest.mark.parametrize('m, k, num_observations', [
    (1000, 2, 10000),
    (1000, 6, 100000),
    (1000, 1, 100000),
])
def test_uniformity(m, k, num_observations):

    hasher = KnuthHash(m, k)

    indices = np.random.randint(0, 10000, size=num_observations)

    collisions = np.zeros(m)

    for idx in indices:
        collisions[hasher.hash(idx)] += 1

    collisions /= k

    _, p_value = scipy.stats.chisquare(collisions)

    assert p_value > 0.1
