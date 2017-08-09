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


def _filter_by_count(elements, min_count):

    unique_elements, element_counts = np.unique(elements,
                                                return_counts=True)

    return unique_elements[element_counts >= min_count]


def _build_contiguous_map(elements):

    return dict(zip(elements, np.arange(len(elements))))


def _map(elements, mapping):

    for idx, elem in enumerate(elements):
        elements[idx] = mapping[elem]

    return elements


def _build_features(feature_item_ids, feature_ids, num_features):

    top_features = _filter_features(feature_ids, num_features)

    retain = np.in1d(feature_ids, top_features)

    feature_item_ids = feature_item_ids[retain]
    feature_ids = feature_ids[retain]

    retain_map = _build_contiguous_map(top_features)
    feature_ids = _map(feature_ids, retain_map)

    features = sp.coo_matrix((np.ones_like(feature_item_ids),
                              (feature_item_ids, feature_ids))).tocsr()

    dense_features = np.squeeze(np.array(features.todense()))

    return dense_features


def get_amazon(min_user_interactions=10, min_item_interactions=10, num_features=1000):
    """
    """

    (user_ids, item_ids, ratings,
     timestamps, feature_item_ids,
     feature_ids) = _download_amazon()

    features = _build_features(feature_item_ids,
                               feature_ids,
                               num_features)

    retain_user_ids = _filter_by_count(user_ids, min_user_interactions)
    retain_item_ids = _filter_by_count(item_ids, min_item_interactions)

    retain = np.logical_and(np.in1d(user_ids, retain_user_ids),
                            np.in1d(item_ids, retain_item_ids))

    user_ids = user_ids[retain]
    item_ids = item_ids[retain]
    ratings = ratings[retain]
    timestamps = timestamps[retain]

    features = features[retain_item_ids]

    retain_user_map = _build_contiguous_map(retain_user_ids)
    retain_item_map = _build_contiguous_map(retain_item_ids)

    user_ids = _map(user_ids, retain_user_map)
    item_ids = _map(item_ids, retain_item_map)

    return Interactions(user_ids,
                        item_ids,
                        ratings=ratings,
                        timestamps=timestamps,
                        item_features=features,
                        num_users=len(retain_user_map),
                        num_items=len(retain_item_map))
