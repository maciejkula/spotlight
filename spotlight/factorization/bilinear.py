import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable


from spotlight.layers import ScaledEmbedding, ZeroEmbedding
from spotlight.torch_utils import cpu, gpu, minibatch


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


class FactorizationModel(object):
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
                 l2=0.0,
                 use_cuda=False,
                 sparse=False):

        assert loss in ('pointwise',
                        'bpr',
                        'adaptive',
                        'regression',
                        'truncated_regression')

        if l2 != 0.0 and sparse:
            raise ValueError('L2 regularization not supported '
                             'with sparse gradients')

        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse

        self._num_users = None
        self._num_items = None
        self._net = None

    def _pointwise_loss(self, users, items, ratings):

        negatives = Variable(
            gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   len(users))),
                self._use_cuda)
        )

        positives_loss = (1.0 - F.sigmoid(self._net(users, items)))
        negatives_loss = F.sigmoid(self._net(users, negatives))

        return torch.cat([positives_loss, negatives_loss]).mean()

    def _bpr_loss(self, users, items, ratings):

        negatives = Variable(
            gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   len(users))),
                self._use_cuda)
        )

        return (1.0 - F.sigmoid(self._net(users, items) -
                                self._net(users, negatives))).mean()

    def _adaptive_loss(self, users, items, ratings):

        negative_predictions = []

        for _ in range(5):
            negatives = Variable(
                gpu(
                    torch.from_numpy(np.random.randint(0,
                                                       self._num_items,
                                                       len(users))),
                    self._use_cuda)
            )

            negative_predictions.append(self._net(users, negatives))

        best_negative_prediction, _ = torch.cat(negative_predictions, 1).max(1)
        positive_prediction = self._net(users, items)

        return torch.mean(torch.clamp(best_negative_prediction -
                                      positive_prediction
                                      + 1.0, 0.0))

    def _truncated_regression_loss(self, users, items, ratings):

        negatives = Variable(
            gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   len(users))),
                self._use_cuda)
        )

        pos_prob, pos_rating, pos_stddev = self._net(users, items)

        positives_likelihood = (torch.log(pos_prob)
                                - 0.5 * torch.log(pos_stddev ** 2)
                                - (0.5 * (pos_rating - ratings) ** 2
                                   / (pos_stddev ** 2)))
        neg_prob, _, _ = self._net(users, negatives)
        negatives_likelihood = torch.log(1.0 - neg_prob)

        return torch.cat([-positives_likelihood, -negatives_likelihood]).mean()

    def _regression_loss(self, users, items, ratings):

        predicted_rating = self._net(users, items)

        return ((ratings - predicted_rating) ** 2).mean()

    def _shuffle(self, interactions):

        users = interactions.row
        items = interactions.col
        ratings = interactions.data

        shuffle_indices = np.arange(len(users))
        np.random.shuffle(shuffle_indices)

        return (users[shuffle_indices].astype(np.int64),
                items[shuffle_indices].astype(np.int64),
                ratings[shuffle_indices].astype(np.float32))

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

        if self._loss in ('truncated_regression',):
            self._net = gpu(
                TruncatedBilinearNet(self._num_users,
                                     self._num_items,
                                     self._embedding_dim,
                                     sparse=self._sparse),
                self._use_cuda
            )
        else:
            self._net = gpu(
                BilinearNet(self._num_users,
                            self._num_items,
                            self._embedding_dim,
                            sparse=self._sparse),
                self._use_cuda
            )

        if self._sparse:
            optimizer = optim.Adagrad(self._net.parameters(),
                                      weight_decay=self._l2)
        else:
            optimizer = optim.Adam(self._net.parameters(),
                                   weight_decay=self._l2)

        if self._loss == 'pointwise':
            loss_fnc = self._pointwise_loss
        elif self._loss == 'bpr':
            loss_fnc = self._bpr_loss
        elif self._loss == 'regression':
            loss_fnc = self._regression_loss
        elif self._loss == 'truncated_regression':
            loss_fnc = self._truncated_regression_loss
        else:
            loss_fnc = self._adaptive_loss

        for epoch_num in range(self._n_iter):

            users, items, ratings = self._shuffle(interactions)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                   self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                   self._use_cuda)
            ratings_tensor = gpu(torch.from_numpy(ratings),
                                  self._use_cuda)

            epoch_loss = 0.0

            for (batch_user,
                 batch_item,
                 batch_ratings) in zip(minibatch(user_ids_tensor,
                                                  self._batch_size),
                                       minibatch(item_ids_tensor,
                                                  self._batch_size),
                                       minibatch(ratings_tensor,
                                                  self._batch_size)):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                ratings_var = Variable(batch_ratings)

                optimizer.zero_grad()

                loss = loss_fnc(user_var, item_var, ratings_var)
                epoch_loss += loss.data[0]

                loss.backward()
                optimizer.step()

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids, ratings=False):
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

        if ratings:
            if self._loss not in ('regression',
                                  'truncated_regression'):
                raise ValueError('Ratings can only be returned '
                                 'when the truncated regression loss '
                                 'is used')

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(gpu(user_ids, self._use_cuda))
        item_var = Variable(gpu(item_ids, self._use_cuda))

        out = self._net(user_var, item_var)

        if self._loss in ('truncated_regression',):
            if ratings:
                return cpu((out[1]).data).numpy().flatten()
            else:
                return cpu((out[0]).data).numpy().flatten()
        else:
            return cpu(out.data).numpy().flatten()


class SequenceModel(object):
    """
    One-ahead prediction model.

    Can use one of the following user representations:
    - pool: pooling over previous items
    - lstm: LSTM over previous items
    - popularity: always predict the most popular item

    Can use one of the following losses
    - pointwise
    - BPR
    - adaptive
    """

    def __init__(self,
                 loss='pointwise',
                 representation='lstm',
                 embedding_dim=64,
                 n_iter=3,
                 batch_size=64,
                 l2=0.0,
                 use_cuda=False,
                 sparse=False):

        assert loss in ('pointwise',
                        'bpr',
                        'adaptive')

        assert representation in ('pool',
                                  'lstm',
                                  'popularity')

        self._loss = loss
        self._representation = representation
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse

        self._num_items = None
        self._net = None

    def _pointwise_loss(self, users, items, ratings):

        negatives = Variable(
            gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   tuple(users.size()))),
                self._use_cuda)
        )

        mask = (items > 0).float()

        positives_loss = (1.0 - F.sigmoid(self._net(users, items))) * mask
        negatives_loss = F.sigmoid(self._net(users, negatives)) * mask

        return torch.cat([positives_loss, negatives_loss]).mean()

    def _bpr_loss(self, users, items, ratings):

        negatives = Variable(
            gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   tuple(users.size()))),
                self._use_cuda)
        )

        mask = (items > 0).float()

        return ((1.0 - F.sigmoid(self._net(users, items) -
                                 self._net(users, negatives))) * mask).mean()

    def _adaptive_loss(self, users, items, ratings):

        negative_predictions = []

        for _ in range(5):
            negatives = Variable(
                gpu(
                    torch.from_numpy(np.random.randint(0,
                                                       self._num_items,
                                                       tuple(users.size()))),
                    self._use_cuda)
            )

            negative_predictions.append(self._net(users, negatives))

        best_negative_prediction, _ = torch.cat(negative_predictions, 2).max(2)
        positive_prediction = self._net(users, items)

        return torch.mean(torch.clamp(best_negative_prediction -
                                      positive_prediction
                                      + 1.0, 0.0))

    def _shuffle(self, sequences, targets):

        shuffle_indices = np.arange(len(targets))
        np.random.shuffle(shuffle_indices)

        return (sequences[shuffle_indices].astype(np.int64),
                targets[shuffle_indices].astype(np.int64))

    def fit(self, sequences, targets, verbose=False):
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

        self._num_items = max(int(sequences.max() + 1),
                              int(targets.max() + 1))

        if self._representation == 'lstm':
            self._net = gpu(
                LSTMNet(self._num_items,
                        self._embedding_dim,
                        sparse=self._sparse),
                self._use_cuda
            )
        elif self._representation == 'popularity':
            self._net = gpu(
                PopularityNet(self._num_items,
                              sparse=self._sparse),
                self._use_cuda
            )
        else:
            self._net = gpu(
                PoolNet(self._num_items,
                        self._embedding_dim,
                        sparse=self._sparse),
                self._use_cuda
            )

        if self._sparse:
            optimizer = optim.Adagrad(self._net.parameters(),
                                      weight_decay=self._l2)
        else:
            optimizer = optim.Adam(self._net.parameters(),
                                   weight_decay=self._l2)

        if self._loss == 'pointwise':
            loss_fnc = self._pointwise_loss
        elif self._loss == 'bpr':
            loss_fnc = self._bpr_loss
        else:
            loss_fnc = self._adaptive_loss

        for epoch_num in range(self._n_iter):

            sequences_tensor = gpu(torch.from_numpy(sequences),
                                    self._use_cuda)
            targets_tensor = gpu(torch.from_numpy(targets),
                                  self._use_cuda)

            epoch_loss = 0.0

            for (batch_user,
                 batch_item) in zip(minibatch(sequences_tensor,
                                               self._batch_size),
                                    minibatch(targets_tensor,
                                               self._batch_size)):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)

                optimizer.zero_grad()

                loss = loss_fnc(user_var, item_var, item_var)
                epoch_loss += loss.data[0]

                loss.backward()
                optimizer.step()

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def compute_mrr(self, sequences, targets, num_samples=20):
        """
        Computes the MRR of one-ahead-prediction among
        a sample of possible candidates.

        Will overestimate true MRR but is a lot faster to compute.
        """

        mask = targets > 0

        sequences = Variable(gpu(torch.from_numpy(sequences.astype(np.int64)),
                                  self._use_cuda), volatile=True)
        targets = Variable(gpu(torch.from_numpy(targets.astype(np.int64)),
                                self._use_cuda),
                           volatile=True)

        positive_scores = self._net(sequences, targets)

        inversion_counts = positive_scores >= positive_scores

        for _ in range(num_samples):

            negatives = Variable(
                gpu(
                    torch.from_numpy(np.random.randint(0,
                                                       self._num_items,
                                                       tuple(targets.size()))),
                    self._use_cuda),
                volatile=True
            )

            negative_scores = self._net(sequences, negatives)

            inversion_counts += negative_scores > positive_scores

        return 1.0 / cpu(inversion_counts.data).numpy().flatten()[mask.flatten()]

    def predict(self, sequences, item_ids):
        """
        Compute the recommendation score for user-item pairs.

        Arguments
        ---------

        item_ids: np.int32 array of shape [n_pairs,]
             an array containing the item ids for the user-item pairs for which
             a prediction is to be computed.
        ratings: bool, optional
             Return predictions on ratings (rather than likelihood of rating)
        """

        sequences = torch.from_numpy(sequences.astype(np.int64))
        targets = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(gpu(sequences, self._use_cuda))
        item_var = Variable(gpu(targets, self._use_cuda))

        out = self._net(user_var, item_var)

        return cpu(out.data).numpy().flatten()
