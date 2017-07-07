"""
Factorization models for implicit feedback problems.
"""

import numpy as np

import torch

import torch.optim as optim

from torch.autograd import Variable


from spotlight.losses import (bpr_loss,
                              hinge_loss,
                              pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.sampling import sample_items
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle


class ImplicitFactorizationModel(object):
    """
    An implict feedback matrix factorization model. Uses a classic
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
                 loss='pointwise',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer=None,
                 use_cuda=False,
                 sparse=False,
                 random_state=None):

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
            The input dataset.
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

        if self._loss == 'pointwise':
            loss_fnc = pointwise_loss
        elif self._loss == 'bpr':
            loss_fnc = bpr_loss
        else:
            loss_fnc = hinge_loss

        for epoch_num in range(self._n_iter):

            users, items = shuffle(user_ids,
                                   item_ids,
                                   random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)

            epoch_loss = 0.0

            for (batch_user,
                 batch_item) in minibatch(user_ids_tensor,
                                          item_ids_tensor,
                                          batch_size=self._batch_size):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                positive_prediction = self._net(user_var, item_var)

                if self._loss == 'adaptive_hinge':
                    negative_prediction = self._get_adaptive_negatives(
                        user_var
                    )
                else:
                    negative_items = sample_items(
                        self._num_items,
                        len(batch_user),
                        random_state=self._random_state)
                    negative_var = Variable(
                        gpu(torch.from_numpy(negative_items))
                    )
                    negative_prediction = self._net(user_var, negative_var)

                self._optimizer.zero_grad()

                loss = loss_fnc(positive_prediction, negative_prediction)
                epoch_loss += loss.data[0]

                loss.backward()
                self._optimizer.step()

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def _get_adaptive_negatives(self, user_ids, num_neg_candidates=5):

        negatives = Variable(
            gpu(
                torch.from_numpy(
                    self._random_state
                    .randint(0, self._num_items,
                             (len(user_ids), num_neg_candidates))),
                self._use_cuda)
        )
        negative_predictions = self._net(
            user_ids.repeat(num_neg_candidates, 1).transpose(0, 1),
            negatives
        ).view(-1, num_neg_candidates)

        best_negative_prediction, _ = negative_predictions.max(1)

        return best_negative_prediction

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

        if item_ids is None:
            item_ids = np.arange(self._num_items)

        if isinstance(user_ids, int):
            user_ids = np.repeat(user_ids, len(item_ids))

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(gpu(user_ids, self._use_cuda))
        item_var = Variable(gpu(item_ids, self._use_cuda))

        out = self._net(user_var, item_var)

        return cpu(out.data).numpy().flatten()
