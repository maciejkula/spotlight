import torch

import torch.nn as nn
import torch.nn.functional as F


from spotlight.layers import ScaledEmbedding, ZeroEmbedding


PADDING_IDX = 0


class PoolNet(nn.Module):

    def __init__(self, num_items, embedding_dim=32, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse,
                                               padding_idx=PADDING_IDX)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

    def user_representation(self, item_sequences):

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

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze(1)

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze(1))

        return target_bias + dot


class LSTMNet(nn.Module):

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

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze(1)

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze(1))

        return target_bias + dot


class CNNNet(nn.Module):

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

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze(1)

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze(1))

        return target_bias + dot
