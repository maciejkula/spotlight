"""
Utilities for fetching Amazon datasets
"""

import h5py

import numpy as np

from spotlight.datasets import _transport
from spotlight.interactions import Interactions


def _download_amazon():

    extension = '.hdf5'
    url = ('https://github.com/maciejkula/recommender_datasets/'
           'releases/download')
    version = '0.1.0'

    path = _transport.get_data('/'.join((url,
                                         version,
                                         'amazon_co_purchasing' + extension)),
                               'amazon',
                               'amazon_co_purchasing{}'.format(extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:],
                data['/features_item_id'][:],
                data['/features_feature_id'][:])


def _filter_by_count(elements, min_count):

    unique_elements, element_counts = np.unique(elements,
                                                return_counts=True)

    return unique_elements[element_counts >= min_count]


def _build_contiguous_map(elements):

    return dict(zip(elements, np.arange(len(elements)) + 1))


def _map(elements, mapping):

    for idx, elem in enumerate(elements):
        elements[idx] = mapping[elem]

    return elements


def get_amazon_dataset(min_user_interactions=10, min_item_interactions=10):
    """
    Data on Amazon products from the SNAP `archive
    <https://snap.stanford.edu/data/amazon-meta.html>`_[1]_.

    The dataset contains almost 8 million ratings given to 550,000 Amazon products:
    interactions represent ratings given to users to products they have reviewed.

    Compared to the Movielens dataset, the Amazon dataset is relatively sparse,
    and the number of products represented is much higher. It may therefore be
    more useful for prototyping models for sparse and high-dimensional settings.

    Parameters
    ----------

    min_user_interactions: int, optional
        Exclude observations from users that have given fewer ratings.
    min_item_interactions: int, optional
        Exclude observations from items that have given fewer ratings.

    Notes
    -----

    You may want to reduce the dimensionality of the dataset by excluding users
    and items with particularly few interactions. Note that the exclusions are
    applied independently, so it is possible for users and items in the remaining
    set to have fewer interactions than specified via the parameters.

    References
    ----------

    .. [1] J. Leskovec, L. Adamic and B. Adamic.
       The Dynamics of Viral Marketing.
       ACM Transactions on the Web (ACM TWEB), 1(1), 2007.
    """

    (user_ids, item_ids, ratings,
     timestamps, feature_item_ids,
     feature_ids) = _download_amazon()

    retain_user_ids = _filter_by_count(user_ids, min_user_interactions)
    retain_item_ids = _filter_by_count(item_ids, min_item_interactions)

    retain = np.logical_and(np.in1d(user_ids, retain_user_ids),
                            np.in1d(item_ids, retain_item_ids))

    user_ids = user_ids[retain]
    item_ids = item_ids[retain]
    ratings = ratings[retain]
    timestamps = timestamps[retain]

    retain_user_map = _build_contiguous_map(retain_user_ids)
    retain_item_map = _build_contiguous_map(retain_item_ids)

    user_ids = _map(user_ids, retain_user_map)
    item_ids = _map(item_ids, retain_item_map)

    return Interactions(user_ids,
                        item_ids,
                        ratings=ratings,
                        timestamps=timestamps,
                        num_users=len(retain_user_map) + 1,
                        num_items=len(retain_item_map) + 1)
