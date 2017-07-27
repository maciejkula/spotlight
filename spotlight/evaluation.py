import numpy as np

import scipy.stats as st


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

    if train is not None:
        train = train.tocsr()

    mrrs = []

    for minibatch in test.minibatches(batch_size=1):

        user_id = int(minibatch.user_ids[0].data.numpy())
        item_id = int(minibatch.item_ids[0].data.numpy())
        predictions = -model.predict(user_id,
                                     user_features=minibatch.user_features,
                                     context_features=minibatch.context_features,
                                     item_features=minibatch.item_features)
        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[item_id])

        mrrs.append(mrr)

    return np.array(mrrs)


def sequence_mrr_score(model, test):
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

        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


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
