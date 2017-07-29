"""
Classes defining user and item latent representations in
factorization models.
"""

import torch.nn as nn
import torch.nn.functional as F

from spotlight.layers import ScaledEmbedding, ZeroEmbedding
from spotlight.torch_utils import concatenate


class HybridContainer(nn.Module):

    def __init__(self, latent_module, context_module=None, item_module=None):

        super(HybridContainer, self).__init__()

        self.latent = latent_module
        self.context = context_module
        self.item = item_module

    def forward(self, user_ids, item_ids,
                user_features=None,
                context_features=None,
                item_features=None):

        user_representation, user_bias = self.latent.user_representation(user_ids)
        item_representation, item_bias = self.latent.item_representation(item_ids)

        if self.context is not None:
            user_representation = self.context(user_representation,
                                               user_features,
                                               context_features)
        if self.item is not None:
            item_representation = self.item(item_representation,
                                            item_features)

        return self.latent(user_representation, user_bias,
                           item_representation, item_bias)


class FeatureNet(nn.Module):

    def __init__(self, input_dim, output_dim, bias=False):

        super(FeatureNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_1 = nn.Linear(self.input_dim,
                              self.output_dim,
                              bias=bias)

    def forward(self, features):

        return self.fc_1(features)


class HybridContextNet(nn.Module):

    def __init__(self, embedding_dim, num_context_features, output_dim):

        super(HybridContextNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_context_features = num_context_features
        self.output_dim = output_dim

        self.fc_1 = nn.Linear(#self.embedding_dim +
                              self.num_context_features,
                              output_dim,
                              bias=False)

    def forward(self, user_representation, user_features, context_features):

        inputs = (user_representation, user_features, context_features)
        x = concatenate(inputs, axis=1)

        # x = F.tanh(self.fc_1(context_features))
        #x = F.tanh(self.fc_1(context_features))
        x = self.fc_1(context_features)

        # assert False

        return x + user_representation
        # return x + user_representation
        # return user_representation
        return user_representation

        # return F.tanh(self.fc_layer(context_features)) + user_representation


class HybridItemNet(nn.Module):

    def __init__(self, embedding_dim, num_item_features, output_dim):

        super(HybridItemNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_item_features = num_item_features
        self.output_dim = output_dim

        self.fc_layer = nn.Linear(self.embedding_dim +
                                  self.num_item_features,
                                  output_dim)

    def forward(self, item_representation, item_features):

        inputs = (item_representation, item_features)
        x = concatenate(inputs, axis=1)

        return self.fc_layer(x)


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
