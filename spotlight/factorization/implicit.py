import numpy as np

import torch

import torch.optim as optim

from torch.autograd import Variable


from spotlight.losses import (bpr_loss,
                              hinge_loss,
                              pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.sampling import sample_items
from spotlight.torch_utils import cpu, gpu, minibatch, shuffle


class ImplicitFactorizationModel(object):
    """
    An implict matrix factorization model.

    Parameters
    ----------

    loss: string, one of 'pointwise', 'bpr', 'hinge', or 'adaptive hinge'
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
