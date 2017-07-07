"""
This module contains prototypes of various ways of representing users
as functions of the items they have interacted with in the past.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F


from spotlight.layers import ScaledEmbedding, ZeroEmbedding


PADDING_IDX = 0


class PoolNet(nn.Module):
    """
    Module representing users through averaging the representations of items
    they have interacted with, a'la [1]_.

    To represent a sequence, it simply averages the representations of all
    the items that occur in the sequence up to that point.

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and aross time in the sequence.

    Parameters
    ----------

    num_items: int
        number of items to be represented
    embedding_dim: int, optional
        embedding dimension of the embedding layer, and the number of filters
        in each convlutonal layer

    References
    ----------

    .. [1] Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for
       youtube recommendations." Proceedings of the 10th ACM Conference
       on Recommender Systems. ACM, 2016.

    """

    def __init__(self, num_items, embedding_dim=32, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse,
                                               padding_idx=PADDING_IDX)
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
            result of the user_representation_method
        targets: tensor
            a minibatch of item sequences of shape
            (minibatch_size, sequence_length)

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze(1)

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze(1))

        return target_bias + dot


class LSTMNet(nn.Module):
    """
    Module representing users through running a recurrent neural network
    over the sequence, using the hidden state at each timestep as the
    sequence representation, a'la [2]_

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and aross time in the sequence.

    Parameters
    ----------

    num_items: int
        number of items to be represented
    embedding_dim: int, optional
        embedding dimension of the embedding layer, and the number of filters
        in each convlutonal layer

    References
    ----------

    .. [2] Hidasi, Balázs, et al. "Session-based recommendations with
       recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
    """

    def __init__(self, num_items, embedding_dim=32, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse,
                                               padding_idx=PADDING_IDX)
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
            result of the user_representation_method
        targets: tensor
            a minibatch of item sequences of shape
            (minibatch_size, sequence_length)

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze(1)

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze(1))

        return target_bias + dot


class CNNNet(nn.Module):
    """
    Module representing users through stacked causal convolutions [3]_.

    To represent a sequence, it runs a 1D convolution over the input sequence,
    from left to right. At each timestep, the output of the convolution is
    the representation of the sequence up to that point. The convoluion is causal
    because future states are never part of the convolution's receptive field;
    this is achieved by left-padding the sequence.

    In order to increase the receptive field (and the capacity to encode states
    further back in the sequence), one can increase the kernel width, or stack
    more layers. Input dimensionality is preserved from layer to layer.

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and aross time in the sequence.

    Parameters
    ----------

    num_items: int
        number of items to be represented
    embedding_dim: int, optional
        embedding dimension of the embedding layer, and the number of filters
        in each convlutonal layer
    kernel_width: int, optional
        the kernel width of the convolutional layers
    num_layers: int, optional
        number of stacked convolutional layers

    References
    ----------

    .. [3] Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio."
       arXiv preprint arXiv:1609.03499 (2016).
    """

    def __init__(self, num_items,
                 embedding_dim=32,
                 kernel_width=5,
                 num_layers=1,
                 sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.kernel_width = kernel_width

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse,
                                               padding_idx=PADDING_IDX)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.cnn_layers = [
            nn.Conv2d(embedding_dim, embedding_dim, (kernel_width, 1)) for
            _ in range(num_layers)
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

        x = sequence_embeddings
        for i, cnn_layer in enumerate(self.cnn_layers):
            # Pad so that the CNN doesn't have the future
            # of the sequence in its receptive field.
            x = F.pad(x, (0, 0, self.kernel_width - min(i, 1), 0))
            x = F.relu(cnn_layer(x))

        x = x.squeeze(3)

        return x[:, :, :-1], x[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            result of the user_representation_method
        targets: tensor
            a minibatch of item sequences of shape
            (minibatch_size, sequence_length)

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze(1)

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze(1))

        return target_bias + dot
