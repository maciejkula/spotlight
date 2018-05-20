import numpy as np
import pytest
import torch

import torch.nn as nn

from spotlight.layers import BloomEmbedding, ScaledEmbedding


@pytest.mark.parametrize('embedding_class', [
    nn.Embedding,
    ScaledEmbedding,
    BloomEmbedding
])
def test_embeddings(embedding_class):

    num_embeddings = 1000
    embedding_dim = 16

    batch_size = 32
    sequence_length = 8

    layer = embedding_class(num_embeddings,
                            embedding_dim)

    # Test 1-d inputs (minibatch)
    indices = torch.from_numpy(
        np.random.randint(0, num_embeddings, size=batch_size, dtype=np.int64))
    representation = layer(indices)
    assert representation.size()[0] == batch_size
    assert representation.size()[-1] == embedding_dim

    # Test 2-d inputs (minibatch x sequence_length)
    indices = torch.from_numpy(
        np.random.randint(0, num_embeddings,
                          size=(batch_size, sequence_length), dtype=np.int64))
    representation = layer(indices)
    assert representation.size() == (batch_size, sequence_length, embedding_dim)
