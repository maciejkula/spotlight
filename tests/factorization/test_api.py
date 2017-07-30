import numpy as np

import pytest

from spotlight.datasets import movielens, synthetic
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel


@pytest.mark.parametrize('model_class', [
    ImplicitFactorizationModel,
    ExplicitFactorizationModel
])
def test_predict_movielens(model_class):

    interactions = movielens.get_movielens_dataset('100K')

    model = model_class(n_iter=1)
    model.fit(interactions)

    for user_id in np.random.randint(0, interactions.num_users, size=10):
        user_ids = np.repeat(user_id, interactions.num_items)
        item_ids = np.arange(interactions.num_items)

        uid_predictions = model.predict(user_id)
        iid_predictions = model.predict(user_id, item_ids)
        pair_predictions = model.predict(user_ids, item_ids)

        assert (uid_predictions == iid_predictions).all()
        assert (uid_predictions == pair_predictions).all()


@pytest.mark.parametrize('model_class', [
    ImplicitFactorizationModel,
    ExplicitFactorizationModel
])
def test_predict_hybrid(model_class):

    interactions = synthetic.generate_content_based(num_interactions=100)

    model = model_class(n_iter=1)
    model.fit(interactions)

    for context in interactions.contexts():
        user_id = context.user_ids[0]
        user_ids = np.repeat(user_id, interactions.num_items)
        item_ids = np.arange(interactions.num_items)

        uid_predictions = model.predict(user_id,
                                        user_features=context.user_features,
                                        context_features=context.context_features,
                                        item_features=context.item_features)
        iid_predictions = model.predict(user_id, item_ids,
                                        user_features=context.user_features,
                                        context_features=context.context_features,
                                        item_features=context.item_features)
        pair_predictions = model.predict(user_ids, item_ids,
                                         user_features=context.user_features,
                                         context_features=context.context_features,
                                         item_features=context.item_features)

        assert (uid_predictions == iid_predictions).all()
        assert (uid_predictions == pair_predictions).all()
