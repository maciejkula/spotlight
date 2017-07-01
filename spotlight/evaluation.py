import numpy as np

import scipy.stats as st


FLOAT_MAX = np.finfo(np.float32).max


def mrr_score(model, test, train=None):

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


def sequence_mrr_score(model, test):

    sequences = test.sequences
    targets = test.targets

    mrrs = []

    for i in range(len(targets)):

        predictions = -model.predict(sequences[i])
        # predictions[sequences[i]] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def rmse_score(model, test):

    predictions = model.predict(test.user_ids, test.item_ids)

    return np.sqrt(((test.ratings - predictions) ** 2).mean())
