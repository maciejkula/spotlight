"""
Factorization models for implicit feedback problems.
"""

import numpy as np

import torch

import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.losses import (adaptive_hinge_loss,
                              bpr_loss,
                              hinge_loss,
                              pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.sampling import sample_items
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle


class ImplicitFactorizationModel(object):
    """
    An implicit feedback matrix factorization model. Uses a classic
    matrix factorization [1]_ approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.

    The latent representation is given by
    :class:`spotlight.factorization.representations.BilinearNet`.

    The model is trained through negative sampling: for any known
    user-item pair, one or more items are randomly sampled to act
    as negatives (expressing a lack of preference by the user for
    the sampled item).

    .. [1] Koren, Yehuda, Robert Bell, and Chris Volinsky.
       "Matrix factorization techniques for recommender systems."
       Computer 42.8 (2009).

    Parameters
    ----------

    loss: string, optional
        One of 'pointwise', 'bpr', 'hinge', or 'adaptive hinge',
        corresponding to losses from :class:`spotlight.losses`.
    embedding_dim: int, optional
        Number of embedding dimensions to use for users and items.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    representation: a representation module, optional
        If supplied, will override default settings and be used as the
        main network module in the model. Intended to be used as an escape
        hatch when you want to reuse the model's training functions but
        want full freedom to specify your network topology.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    num_negative_samples: int, optional
        Number of negative samples to generate for adaptive hinge loss.
    """

    def __init__(self,
                 loss='pointwise',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer_func=None,
                 use_cuda=False,
                 representation=None,
                 sparse=False,
                 random_state=None,
                 num_negative_samples=5):

        assert loss in ('pointwise',
                        'bpr',
                        'hinge',
                        'adaptive_hinge')

        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._representation = representation
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()
        self._num_negative_samples = num_negative_samples

        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None

        set_seed(self._random_state.randint(-10**8, 10**8),
                 cuda=self._use_cuda)

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

        if self._representation is not None:
            self._net = gpu(self._representation,
                            self._use_cuda)
        else:
            self._net = gpu(
                BilinearNet(self._num_users,
                            self._num_items,
                            self._embedding_dim,
                            sparse=self._sparse),
                self._use_cuda
            )

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        if self._loss == 'pointwise':
            self._loss_func = pointwise_loss
        elif self._loss == 'bpr':
            self._loss_func = bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = hinge_loss
        else:
            self._loss_func = adaptive_hinge_loss

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, interactions, verbose=False):
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset.

        verbose: bool
            Output additional information about current epoch and loss.
        """

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        if not self._initialized:
            self._initialize(interactions)

        self._check_input(user_ids, item_ids)

        for epoch_num in range(self._n_iter):

            users, items = shuffle(user_ids,
                                   item_ids,
                                   random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)

            epoch_loss = 0.0

            for (minibatch_num,
                 (batch_user,
                  batch_item)) in enumerate(minibatch(user_ids_tensor,
                                                      item_ids_tensor,
                                                      batch_size=self._batch_size)):

                positive_prediction = self._net(batch_user, batch_item)

                if self._loss == 'adaptive_hinge':
                    negative_prediction = self._get_multiple_negative_predictions(
                        batch_user, n=self._num_negative_samples)
                else:
                    negative_prediction = self._get_negative_prediction(batch_user)

                self._optimizer.zero_grad()

                loss = self._loss_func(positive_prediction, negative_prediction)
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))

    def _get_negative_prediction(self, user_ids):

        negative_items = sample_items(
            self._num_items,
            len(user_ids),
            random_state=self._random_state)
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)

        negative_prediction = self._net(user_ids, negative_var)

        return negative_prediction

    def _get_multiple_negative_predictions(self, user_ids, n=5):

        batch_size = user_ids.size(0)

        negative_prediction = self._get_negative_prediction(user_ids
                                                            .view(batch_size, 1)
                                                            .expand(batch_size, n)
                                                            .reshape(batch_size * n))

        return negative_prediction.view(n, len(user_ids))

    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Parameters
        ----------

        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
        """

        self._check_input(user_ids, item_ids, allow_items_none=True)
        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)

        out = self._net(user_ids, item_ids)

        return cpu(out).detach().numpy().flatten()
