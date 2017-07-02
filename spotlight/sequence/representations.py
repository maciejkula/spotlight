import torch.nn as nn


from spotlight.layers import ScaledEmbedding, ZeroEmbedding


PADDING_IDX = 0


class PoolNet(nn.Module):

    def __init__(self, num_items, embedding_dim, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse,
                                               padding_idx=PADDING_IDX)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

    def user_representation(self, item_sequences):

        sequence_embeddings = self.item_embeddings(item_sequences)

        # Average representations, ignoring padding.

        sequence_embedding_sum = (sequence_embeddings
                                  .sum(1)
                                  .view(item_sequences.size()[0], -1))
        non_padding_entries = ((item_sequences != PADDING_IDX)
                               .float()
                               .sum(1)
                               .expand_as(sequence_embedding_sum))

        user_representations = (
            sequence_embedding_sum / (non_padding_entries + 1)
        )

        return user_representations

    def forward(self, user_representations, targets):

        target_embedding = self.item_embeddings(targets)
        target_bias = self.item_biases(targets)

        dot = (user_representations * target_embedding).sum(1)

        return target_bias + dot


class LSTMNet(nn.Module):

    def __init__(self, num_items, embedding_dim, sparse=False):
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

        sequence_embeddings = self.item_embeddings(item_sequences)

        user_representations, (hidden, cell) = self.lstm(sequence_embeddings)

        return hidden.view(-1, self.embedding_dim)

        return user_representations[:, -1, :]

        return user_representations

    def forward(self, user_representations, targets):

        target_embedding = self.item_embeddings(targets)
        target_bias = self.item_biases(targets)

        dot = (user_representations * target_embedding).sum(1)

        return target_bias + dot


class CNNNet(nn.Module):

    def __init__(self, num_items,
                 embedding_dim,
                 kernel_width=5,
                 num_layers=2,
                 sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse,
                                               padding_idx=PADDING_IDX)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.cnn_layers = [
            nn.Conv2d(embedding_dim, embedding_dim, (kernel_width, 1)) for
            _ in range(num_layers)
        ]

    def user_representation(self, item_sequences):

        sequence_embeddings = self.item_embeddings(item_sequences)

        (batch_size, seq_len, dim) = sequence_embeddings.size()

        # Move embedding dimensions to channels and add a fourth dim.
        sequence_embeddings = (sequence_embeddings
                               .permute(0, 2, 1)
                               .contiguous()
                               .view(batch_size, dim, seq_len, 1))

        x = sequence_embeddings
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)

        user_representations = x.view(batch_size, dim, -1)
        pooled_representations = (user_representations
                                  .max(-1)[0]
                                  .view(batch_size, dim))

        return pooled_representations

    def forward(self, user_representations, targets):

        target_embedding = self.item_embeddings(targets)
        target_bias = self.item_biases(targets)

        dot = (user_representations * target_embedding).sum(1)

        return target_bias + dot
