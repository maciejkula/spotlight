"""
Classes defining user and item latent representations in
factorization models.
"""

import torch.nn as nn
import torch.nn.functional as F

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


class HybridContainer(nn.Module):

    def __init__(self,
                 latent_module,
                 user_module=None,
                 context_module=None,
                 item_module=None):

        super(HybridContainer, self).__init__()

        self.latent = latent_module
        self.user = user_module
        self.context = context_module
        self.item = item_module

    def forward(self, user_ids,
                item_ids,
                user_features=None,
                context_features=None,
                item_features=None):

        user_representation, user_bias = self.latent.user_representation(user_ids)
        item_representation, item_bias = self.latent.item_representation(item_ids)

        if self.user is not None:
            user_representation += self.user(user_features)
        if self.context is not None:
            user_representation += self.context(context_features)
        if self.item is not None:
            item_representation += self.item(item_features)

        dot = (user_representation * item_representation).sum(1)

        return dot + user_bias + item_bias


class FeatureNet(nn.Module):

    def __init__(self, input_dim, output_dim, bias=False, nonlinearity='tanh'):

        super(FeatureNet, self).__init__()

        if nonlinearity == 'tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = F.relu
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = F.sigmoid
        elif nonlinearity == 'linear':
            self.nonlinearity = lambda x: x
        else:
            raise ValueError('Nonlineariy must be one of '
                             '(tanh, relu, sigmoid, linear)')

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_1 = nn.Linear(self.input_dim,
                              self.output_dim,
                              bias=bias)

    def forward(self, features):

        return self.nonlinearity(self.fc_1(features))


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, embedding_dim=32, sparse=False):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                               sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def user_representation(self, user_ids):

        user_embedding = self.user_embeddings(user_ids)
        user_embedding = user_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)

        return user_embedding, user_bias

    def item_representation(self, item_ids):

        item_embedding = self.item_embeddings(item_ids)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        item_bias = self.item_biases(item_ids).view(-1, 1)

        return item_embedding, item_bias

    def forward(self, user_representation, user_bias, item_representation, item_bias):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        dot = (user_representation * item_representation).sum(1)

        return dot + user_bias + item_bias
