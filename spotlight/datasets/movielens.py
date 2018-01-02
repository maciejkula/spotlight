"""
Utilities for fetching the Movielens datasets [1]_.

References
----------

.. [1] https://grouplens.org/datasets/movielens/
"""

import os

import h5py

from spotlight.datasets import _transport
from spotlight.interactions import Interactions

VARIANTS = ('100K',
            '1M',
            '10M',
            '20M')


URL_PREFIX = ('https://github.com/maciejkula/recommender_datasets/'
              'releases/download')
VERSION = 'v0.2.0'


def _get_movielens(dataset):

    extension = '.hdf5'

    path = _transport.get_data('/'.join((URL_PREFIX,
                                         VERSION,
                                         dataset + extension)),
                               os.path.join('movielens', VERSION),
                               'movielens_{}{}'.format(dataset,
                                                       extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:])


def get_movielens_dataset(variant='100K'):
    """
    Download and return one of the Movielens datasets.

    Parameters
    ----------

    variant: string, optional
         String specifying which of the Movielens datasets
         to download. One of ('100K', '1M', '10M', '20M').

    Returns
    -------

    Interactions: :class:`spotlight.interactions.Interactions`
        instance of the interactions class
    """

    if variant not in VARIANTS:
        raise ValueError('Variant must be one of {}, '
                         'got {}.'.format(VARIANTS, variant))

    url = 'movielens_{}'.format(variant)

    return Interactions(*_get_movielens(url))
