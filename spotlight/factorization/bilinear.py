import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable


from spotlight.layers import ScaledEmbedding, ZeroEmbedding
from spotlight.losses import (bpr_loss,
                              hinge_loss,
                              pointwise_loss)
from spotlight.torch_utils import cpu, gpu, minibatch, shuffle


class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                               sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.view(-1, self.embedding_dim)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)
        item_bias = self.item_biases(item_ids).view(-1, 1)

        dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias


class TruncatedBilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.rating_net = BilinearNet(num_users, num_items,
                                      embedding_dim, sparse=sparse)
        self.observed_net = BilinearNet(num_users, num_items,
                                        embedding_dim, sparse=sparse)

        self.stddev = nn.Embedding(1, 1)

    def forward(self, user_ids, item_ids):

        observed = F.sigmoid(self.observed_net(user_ids, item_ids))
        rating = self.rating_net(user_ids, item_ids)
        stddev = self.stddev((user_ids < -1).long()).view(-1, 1)

        return observed, rating, stddev


class ImplicitFactorizationModel(object):
    """
    A number of classic factorization models, implemented in PyTorch.

    Available loss functions:
    - pointwise logistic
    - BPR: Rendle's personalized Bayesian ranking
    - adaptive: a variant of WARP with adaptive selection of negative samples
    - regression: minimizing the regression loss between true and predicted ratings
    - truncated_regression: truncated regression model, that jointly models
                            the likelihood of a rating being given and the value
                            of the rating itself.

    Performance notes: neural network toolkits do not perform well on sparse tasks
    like recommendations. To achieve acceptable speed, either use the `sparse` option
    on a CPU or use CUDA with very big minibatches (1024+).
    """

    def __init__(self,
                 loss='pointwise',
                 embedding_dim=64,
                 n_iter=3,
                 batch_size=64,
                 optimizer=None,
                 use_cuda=False,
                 sparse=False):

        assert loss in ('pointwise',
                        'bpr',
                        'adaptive')

        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._optimizer = None

        self._num_users = None
        self._num_items = None
        self._net = None

    def fit(self, interactions, verbose=False):
        """
        Fit the model.

        Arguments
        ---------

        interactions: np.float32 coo_matrix of shape [n_users, n_items]
             the matrix containing
             user-item interactions. The entries can be binary
             (for implicit tasks) or ratings (for regression
             and truncated regression).
        verbose: Bool, optional
             Whether to print epoch loss statistics.
        """

        self._num_users, self._num_items = interactions.shape

        self._net = gpu(
            BilinearNet(self._num_users,
                        self._num_items,
                        self._embedding_dim,
                        sparse=self._sparse),
            self._use_cuda
        )

        if self._optimizer is None:
            self._optimizer = optim.Adam(self._net.parameters())

        if self._loss == 'pointwise':
            loss_fnc = pointwise_loss
        elif self._loss == 'bpr':
            loss_fnc = bpr_loss
        elif self._loss == 'hinge':
            loss_fnc = hinge_loss

        for epoch_num in range(self._n_iter):

            users, items, ratings = shuffle(*(interactions.row,
                                              interactions.col,
                                              interactions.data))

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)

            epoch_loss = 0.0

            for (batch_user,
                 batch_item,
                 batch_ratings) in minibatch(user_ids_tensor,
                                             item_ids_tensor,
                                             batch_size=self._batch_size):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                ratings_var = Variable(batch_ratings)

                self._optimizer.zero_grad()

                loss = loss_fnc(user_var, item_var, ratings_var)
                epoch_loss += loss.data[0]

                loss.backward()
                self._optimizer.step()

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids):
        """
        Compute the recommendation score for user-item pairs.

        Arguments
        ---------

        user_ids: integer or np.int32 array of shape [n_pairs,]
             single user id or an array containing the user ids for the user-item pairs for which
             a prediction is to be computed
        item_ids: np.int32 array of shape [n_pairs,]
             an array containing the item ids for the user-item pairs for which
             a prediction is to be computed.
        ratings: bool, optional
             Return predictions on ratings (rather than likelihood of rating)
        """

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(gpu(user_ids, self._use_cuda))
        item_var = Variable(gpu(item_ids, self._use_cuda))

        out = self._net(user_var, item_var)

        return cpu(out.data).numpy().flatten()
