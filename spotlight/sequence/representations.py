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
