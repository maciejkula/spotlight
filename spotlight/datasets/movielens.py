import os

import h5py

from spotlight.datasets import _transport
from spotlight.interactions import Interactions

VARIANTS = ('100K',
            '1M',
            '10M',
            '20M')


URL_PREFIX = ('https://github.com/maciejkula/recommender_datasets/'
              'releases/download/')
VERSION = '0.1.0'


def _get_movielens(dataset):

    extension = '.hdf5'

    path = _transport.get_data(os.path.join(URL_PREFIX,
                                            VERSION,
                                            dataset + extension),
                               'movielens',
                               'movielens_{}{}'.format(dataset,
                                                       extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:])


def get_movielens_dataset(variant='100K'):

    if variant not in VARIANTS:
        raise ValueError('Variant must be one of {}, '
                         'got {}.'.format(VARIANTS, variant))

    url = 'movielens_{}'.format(variant)

    return Interactions(*_get_movielens(url))
