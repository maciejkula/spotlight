"""
This module contains prototypes of various ways of representing users
as functions of the items they have interacted with in the past.
"""

import torch

from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


PADDING_IDX = 0


def _to_iterable(val, num):

    try:
        iter(val)
        return val
    except TypeError:
        return (val,) * num


class PoolNet(nn.Module):
    """
    Module representing users through averaging the representations of items
    they have interacted with, a'la [1]_.

    To represent a sequence, it simply averages the representations of all
    the items that occur in the sequence up to that point.

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [1] Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for
       youtube recommendations." Proceedings of the 10th ACM Conference
       on Recommender Systems. ACM, 2016.

    """

    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(PoolNet, self).__init__()

        self.embedding_dim = embedding_dim

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))

        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))

        # Pad it with zeros from left
        sequence_embeddings = F.pad(sequence_embeddings,
                                    (0, 0, 1, 0))

        # Average representations, ignoring padding.
        sequence_embedding_sum = torch.cumsum(sequence_embeddings, 2)
        non_padding_entries = (
            torch.cumsum((sequence_embeddings != 0.0).float(), 2)
            .expand_as(sequence_embedding_sum)
        )

        user_representations = (
            sequence_embedding_sum / (non_padding_entries + 1)
        ).squeeze(3)

        return user_representations[:, :, :-1], user_representations[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            Minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1))

        return target_bias + dot


class LSTMNet(nn.Module):
    """
    Module representing users through running a recurrent neural network
    over the sequence, using the hidden state at each timestep as the
    sequence representation, a'la [2]_

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [2] Hidasi, Balazs, et al. "Session-based recommendations with
       recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
    """

    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(LSTMNet, self).__init__()

        self.embedding_dim = embedding_dim

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)

        return user_representations[:, :, :-1], user_representations[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            A minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class CNNNet(nn.Module):
    """
    Module representing users through stacked causal atrous convolutions ([3]_, [4]_).

    To represent a sequence, it runs a 1D convolution over the input sequence,
    from left to right. At each timestep, the output of the convolution is
    the representation of the sequence up to that point. The convolution is causal
    because future states are never part of the convolution's receptive field;
    this is achieved by left-padding the sequence.

    In order to increase the receptive field (and the capacity to encode states
    further back in the sequence), one can increase the kernel width, stack
    more layers, or increase the dilation factor.
    Input dimensionality is preserved from layer to layer.

    Residual connections can be added between all layers.

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of filters
        in each convolutional layer.
    kernel_width: tuple or int, optional
        The kernel width of the convolutional layers. If tuple, should contain
        the kernel widths for all convolutional layers. If int, it will be
        expanded into a tuple to match the number of layers.
    dilation: tuple or int, optional
        The dilation factor for atrous convolutions. Setting this to a number
        greater than 1 inserts gaps into the convolutional layers, increasing
        their receptive field without increasing the number of parameters.
        If tuple, should contain the dilation factors for all convolutional
        layers. If int, it will be expanded into a tuple to match the number
        of layers.
    num_layers: int, optional
        Number of stacked convolutional layers.
    nonlinearity: string, optional
        One of ('tanh', 'relu'). Denotes the type of non-linearity to apply
        after each convolutional layer.
    residual_connections: boolean, optional
        Whether to use residual connections between convolutional layers.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [3] Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio."
       arXiv preprint arXiv:1609.03499 (2016).
    .. [4] Kalchbrenner, Nal, et al. "Neural machine translation in linear time."
       arXiv preprint arXiv:1610.10099 (2016).
    """

    def __init__(self, num_items,
                 embedding_dim=32,
                 kernel_width=3,
                 dilation=1,
                 num_layers=1,
                 nonlinearity='tanh',
                 residual_connections=True,
                 sparse=False,
                 benchmark=True,
                 item_embedding_layer=None):

        super(CNNNet, self).__init__()

        cudnn.benchmark = benchmark

        self.embedding_dim = embedding_dim
        self.kernel_width = _to_iterable(kernel_width, num_layers)
        self.dilation = _to_iterable(dilation, num_layers)
        if nonlinearity == 'tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = F.relu
        else:
            raise ValueError('Nonlinearity must be one of (tanh, relu)')
        self.residual_connections = residual_connections

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.cnn_layers = [
            nn.Conv2d(embedding_dim,
                      embedding_dim,
                      (_kernel_width, 1),
                      dilation=(_dilation, 1)) for
            (_kernel_width, _dilation) in zip(self.kernel_width,
                                              self.dilation)
        ]

        for i, layer in enumerate(self.cnn_layers):
            self.add_module('cnn_{}'.format(i),
                            layer)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))

        # Pad so that the CNN doesn't have the future
        # of the sequence in its receptive field.
        receptive_field_width = (self.kernel_width[0] +
                                 (self.kernel_width[0] - 1) *
                                 (self.dilation[0] - 1))

        x = F.pad(sequence_embeddings,
                  (0, 0, receptive_field_width, 0))
        x = self.nonlinearity(self.cnn_layers[0](x))

        if self.residual_connections:
            residual = F.pad(sequence_embeddings,
                             (0, 0, 1, 0))
            x = x + residual

        for (cnn_layer, kernel_width, dilation) in zip(self.cnn_layers[1:],
                                                       self.kernel_width[1:],
                                                       self.dilation[1:]):
            receptive_field_width = (kernel_width +
                                     (kernel_width - 1) *
                                     (dilation - 1))
            residual = x
            x = F.pad(x, (0, 0, receptive_field_width - 1, 0))
            x = self.nonlinearity(cnn_layer(x))

            if self.residual_connections:
                x = x + residual

        x = x.squeeze(3)

        return x[:, :, :-1], x[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            Minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            Of shape (minibatch_size, sequence_length).
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class MixtureLSTMNet(nn.Module):
    """
    A representation that models users as mixtures-of-tastes.

    This is accomplished via an LSTM with a layer on top that
    projects the last hidden state taste vectors and
    taste attention vectors that match items with the taste
    vectors that are best for evaluating them.

    For a full description of the model, see [5]_.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    num_mixtures: int, optional
        Number of mixture components (distinct user tastes) that
        the network should model.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [5] Kula, Maciej. "Mixture-of-tastes Models for Representing
       Users with Diverse Interests" https://github.com/maciejkula/mixture (2017)
    """

    def __init__(self, num_items,
                 embedding_dim=32,
                 num_mixtures=4,
                 item_embedding_layer=None,
                 sparse=False):

        super(MixtureLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_mixtures = num_mixtures

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)
        self.projection = nn.Conv1d(embedding_dim,
                                    embedding_dim * self.num_mixtures * 2,
                                    kernel_size=1)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """

        batch_size, sequence_length = item_sequences.size()

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        user_representations = self.projection(user_representations)
        user_representations = user_representations.view(batch_size,
                                                         self.num_mixtures * 2,
                                                         self.embedding_dim,
                                                         sequence_length + 1)

        return user_representations[:, :, :, :-1], user_representations[:, :, :, -1:]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            A minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        user_components = user_representations[:, :self.num_mixtures, :, :]
        mixture_vectors = user_representations[:, self.num_mixtures:, :, :]

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze()

        mixture_weights = (mixture_vectors * target_embedding
                           .unsqueeze(1)
                           .expand_as(user_components))
        mixture_weights = (F.softmax(mixture_weights.sum(2), 1)
                           .unsqueeze(2)
                           .expand_as(user_components))
        weighted_user_representations = (mixture_weights * user_components).sum(1)

        dot = ((weighted_user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot
