import torch.nn as nn

import torch.nn.functional as F


from spotlight.layers import ScaledEmbedding, ZeroEmbedding


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
