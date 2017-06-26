import numpy as np

import scipy.stats as st


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
            predictions[train[user_id].indices] = np.finfo(np.float32).max

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)
