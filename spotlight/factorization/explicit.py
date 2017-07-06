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
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids):
        """
        """

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(gpu(user_ids, self._use_cuda))
        item_var = Variable(gpu(item_ids, self._use_cuda))

        out = self._net(user_var, item_var)

        if self._loss == 'poisson':
            out = torch.exp(out)

        return cpu(out.data).numpy().flatten()
