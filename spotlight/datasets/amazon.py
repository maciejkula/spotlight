"""
Utilities for fetching Amazon datasets
"""

import os

import h5py
import numpy as np
import scipy.sparse as sp

from spotlight.datasets import _transport
from spotlight.interactions import Interactions


def _download_amazon():

    extension = '.hdf5'
    url = ('https://github.com/maciejkula/recommender_datasets/'
           'releases/download/')
    version = '0.1.0'

    path = _transport.get_data(os.path.join(url,
                                            version,
                                            'amazon_co_purchasing' + extension),
                               'amazon',
                               'amazon_co_purchasing{}'.format(extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:],
                data['/features_item_id'][:],
                data['/features_feature_id'][:])


def _filter_features(feature_ids, num_features):

    unique_features, feature_counts = np.unique(feature_ids,
                                                return_counts=True)
    top_features = np.argsort(-feature_counts)[:num_features]

    return unique_features[top_features]


def get_amazon(num_features=1000):
    """
    """

    (user_ids, item_ids, ratings,
     timestamps, feature_item_ids,
     feature_ids) = _download_amazon()

    top_features = _filter_features(feature_ids, num_features)

    retain = np.in1d(feature_ids, top_features)

    feature_item_ids = feature_item_ids[retain]
    feature_ids = feature_ids[retain]

    # Translate features to a contiguous range
    feature_dict = {}
    for idx, fidx in enumerate(feature_ids):
        feature_ids[idx] = feature_dict.setdefault(fidx, len(feature_dict))

    features = sp.coo_matrix((np.ones_like(feature_item_ids),
                              (feature_item_ids, feature_ids))).tocsr()

    dense_features = features.todense()

    return Interactions(user_ids,
                        item_ids,
                        ratings=ratings,
                        timestamps=timestamps,
                        item_features=dense_features)
