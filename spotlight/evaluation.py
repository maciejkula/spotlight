import numpy as np

import scipy.stats as st

from sklearn.metrics.pairwise import cosine_similarity

import random


FLOAT_MAX = np.finfo(np.float32).max


def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores. One score
    is given for every user with interactions in the test
    set, representing the mean reciprocal rank of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will be set to very low values and so not
        affect the MRR.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each user in test.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    mrrs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def sequence_mrr_score(model, test, exclude_preceding=False):
    """
    Compute mean reciprocal rank (MRR) scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last elements, is used to predict the last element.

    The reciprocal rank of the last element is returned for each
    sequence.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """

    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]

    mrrs = []

    for i in range(len(sequences)):

        predictions = -model.predict(sequences[i])

        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / len(predictions), float(num_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or array of int,
        The maximum number of predicted items
    Returns
    -------

    (Precision@k, Recall@k): numpy array of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall


def rmse_score(model, test):
    """
    Compute RMSE score for test interactions.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.

    Returns
    -------

    rmse_score: float
        The RMSE score.
    """

    predictions = model.predict(test.user_ids, test.item_ids)

    return np.sqrt(((test.ratings - predictions) ** 2).mean())


def intra_distance_score(model, test, train, k=10,
                         f_distance=lambda x, y: 1 - cosine_similarity([x, y])[0][1],
                         percentage=1):
    """
    Compute IntraDistance@k score aka individual diversity.
    The intra-list diversity of a set of recommended items is defined
    as the average pairwise distance of the items in the set.

    Each item is represented as a vector with length of # of users in train set.
    Values of the vectors are ratings from the users to the particular items.
    For instance; in the following data set (user_id, item_id, rating)
    1, 1, 4
    2, 1, 3
    3, 2, 5
    the item vector for item = 1 is;
    [4, 3, 0]

    Item vectors are cached.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or array of int,
        The maximum number of predicted items
    f_distance: distance function. it measures distance between
        two items. Default value is 1 - cosine similarity
    percentage: Percentage of users to be evaluated.
        Values between >0 and 1 are applicable.
        Default value is 1.
    Returns
    -------

    (IntraDistance@k): numpy array of shape (#evaluated_users, len(k * (k-1) / 2)
        A list of distances between each item in recommendation
        list with length k for each user.
    """

    distances = []
    test = test.tocsr()
    train = train.tocoo()
    cache = {}
    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        if random.uniform(0, 1) > percentage:
            continue

        predictions = -model.predict(user_id)
        rec_list = predictions.argsort()[:k]
        distance = [
            f_distance(_get_item_vector(first_item, train, cache),
                       _get_item_vector(second_item, train, cache))
            for i, first_item in enumerate(rec_list)
            for second_item in rec_list[(i + 1):]
        ]
        distances.append(distance)
    return distances


def _get_item_vector(item_id, train, cache):
    users = train.row
    items = train.col
    data = train.data
    if item_id in cache:
        first_item_vector = cache[item_id]
    else:
        first_item_vector = np.zeros(max(users) + 1)
        for k, x in enumerate(data):
            if items[k] == item_id:
                first_item_vector[users[k]] = x
        cache[item_id] = first_item_vector
    return first_item_vector
