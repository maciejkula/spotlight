"""
Factorization models for explicit feedback problems.
"""

import numpy as np

import torch

import torch.optim as optim

from torch.autograd import Variable


from spotlight.factorization.representations import BilinearNet

from spotlight.losses import (regression_loss,
                              poisson_loss)

from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle


class ExplicitFactorizationModel(object):
    """
    An explicit feedback matrix factorization model. Uses a classic
    matrix factorization [1]_ approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.

    The latent representation is given by
    :class:`spotlight.factorization.representations.BilinearNet`.

    .. [1] Koren, Yehuda, Robert Bell, and Chris Volinsky.
       "Matrix factorization techniques for recommender systems."
       Computer 42.8 (2009).

    Parameters
    ----------

    loss: string, optional
        One of 'regression', 'poisson',
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
    optimizer: instance of a PyTorch optimizer, optional
        Overrides l2 and learning rate if supplied.
    use_cuda: boolean, optional
        Run the model on a GPU.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    """

    def __init__(self,
                 loss='regression',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer=None,
                 use_cuda=False,
                 sparse=False,
                 random_state=None):

        assert loss in ('regression',
                        'poisson')

        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._optimizer = None
        self._random_state = random_state or np.random.RandomState()

        self._num_users = None
        self._num_items = None
        self._net = None

        set_seed(self._random_state.randint(-10**8, 10**8),
                 cuda=self._use_cuda)

    def fit(self, interactions, verbose=False):
        """
        Fit the model.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset. Must have ratings.
        """

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

        self._net = gpu(
            BilinearNet(self._num_users,
                        self._num_items,
                        self._embedding_dim,
                        sparse=self._sparse),
            self._use_cuda
        )

        if self._optimizer is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )

        if self._loss == 'regression':
            loss_fnc = regression_loss
        elif self._loss == 'poisson':
            loss_fnc = poisson_loss
        else:
            raise ValueError('Unknown loss: {}'.format(self._loss))

        for epoch_num in range(self._n_iter):

            users, items, ratings = shuffle(user_ids,
                                            item_ids,
                                            interactions.ratings,
                                            random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)
            ratings_tensor = gpu(torch.from_numpy(ratings),
                                 self._use_cuda)

            epoch_loss = 0.0

            for (batch_user,
                 batch_item,
                 batch_ratings) in minibatch(user_ids_tensor,
                                             item_ids_tensor,
                                             ratings_tensor,
                                             batch_size=self._batch_size):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                ratings_var = Variable(batch_ratings)

                predictions = self._net(user_var, item_var)

                if self._loss == 'poisson':
                    predictions = torch.exp(predictions)

                self._optimizer.zero_grad()

                loss = loss_fnc(ratings_var, predictions)
                epoch_loss += loss.data[0]

                loss.backward()
                self._optimizer.step()

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num,
                                                 epoch_loss / (epoch_num + 1)))

    def predict(self, user_ids, item_ids):
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

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(gpu(user_ids, self._use_cuda))
        item_var = Variable(gpu(item_ids, self._use_cuda))

        out = self._net(user_var, item_var)

        if self._loss == 'poisson':
            out = torch.exp(out)

        return cpu(out.data).numpy().flatten()
