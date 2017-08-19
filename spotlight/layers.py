"""
Embedding layers useful for recommender models.
"""

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable


PRIMES = [
    179424941, 179425457, 179425907, 179426369,
    179424977, 179425517, 179425943, 179426407,
    179424989, 179425529, 179425993, 179426447,
    179425003, 179425537, 179426003, 179426453,
    179425019, 179425559, 179426029, 179426491,
    179425027, 179425579, 179426081, 179426549
]


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the emedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the emedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class BloomEmbedding(nn.Module):
    """
    An embedding layer that compresses the number of embedding
    parameters required by using bloom filter-like hashing.

    Parameters
    ----------

    num_embeddings: int
        Number of entities to be represented.
    embedding_dim: int
        Latent dimension of the embedding.
    bag: boolean, optional
        Whether to use the EmbeddingBag layer.
        Faster, but not available for sequence problems.
    compression_ratio: float, optional
        The underlying number of rows in the embedding layer
        after compression. Numbers below 1.0 will use more
        and more compression, reducing the number of parameters
        in the layer.
    num_hash_functions: int, optional
        Number of hash functions used to compute the bloom filter indices.

    Notes
    -----

    Large embedding layers are a performance problem for fitting models:
    even though the gradients are sparse (only a handful of user and item
    vectors need parameter updates in every minibatch), PyTorch updates
    the entire embedding layer at every backward pass. Computation time
    is then wasted on applying zero gradient steps to whole embedding matrix.

    To alleviate this problem, we can use a smaller underlying embedding layer,
    and probabilistically hash users and items into that smaller space. With
    good hash functions, collisions should be rare, and we should observe
    fitting speedups without a decrease in accuracy.

    The idea follows the RecSys 2017 "Getting recommenders fit"[1]_
    paper. The authors use a bloom-filter-like approach to hashing. Their approach
    uses one-hot encoded inputs followed by fully connected layers as
    well as softmax layers for the output, and their hashing reduces the
    size of the fully connected layers rather than embedding layers as
    implemented here; mathematically, however, the two formulations are
    identical.

    The hash function used is simple multiplicative hashing with a
    different prime for every hash function, modulo the size of the
    compressed embedding layer.

    References
    ----------

    .. [1] Serra, Joan, and Alexandros Karatzoglou.
       "Getting deep recommenders fit: Bloom embeddings
       for sparse binary input/output networks."
       arXiv preprint arXiv:1706.03993 (2017).
    """

    def __init__(self, num_embeddings, embedding_dim,
                 compression_ratio=0.6,
                 num_hash_functions=2,
                 padding_idx=None):

        super(BloomEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression_ratio = compression_ratio
        self.compressed_num_embeddings = int(compression_ratio *
                                             num_embeddings)
        self.num_hash_functions = num_hash_functions

        if num_hash_functions > len(PRIMES):
            raise ValueError('Can use at most {} hash functions ({} requested)'
                             .format(len(PRIMES), num_hash_functions))

        self._masks = PRIMES[:self.num_hash_functions]

        self.embeddings = ScaledEmbedding(self.compressed_num_embeddings,
                                          self.embedding_dim,
                                          padding_idx=padding_idx)

        # Caches for output tensors
        self._masks_tensor = None
        self._indices_cache = None
        self._sequence_cache = None

    def __repr__(self):

        return ('<BloomEmbedding (compression_ratio: {}): {}>'
                .format(self.compression_ratio,
                        repr(self.embeddings)))

    def _initialize_caches(self, indices):

        masks_size = indices.size() + (len(self._masks),)

        if (self._masks_tensor is None or
                self._masks_tensor.size() != masks_size):

            masks = (torch
                     .from_numpy(np.array(self._masks, dtype=np.int64))
                     .expand(masks_size))

            self._masks_tensor = masks
            self._indices_cache = masks * 0

        return self._masks_tensor, self._indices_cache

    def forward(self, indices):
        """
        Retrieve embeddings corresponding to indices.

        See documentation on PyTorch ``nn.Embedding`` for details.
        """

        (masks,
         masked_indices) = self._initialize_caches(indices)

        torch.mul(
            indices.data.unsqueeze(indices.dim()).expand_as(masks),
            masks,
            out=masked_indices)

        masked_indices.remainder_(self.compressed_num_embeddings)
        masked_indices = Variable(masked_indices)

        if masked_indices.dim() == 2:
            embedding = self.embeddings(masked_indices).mean(1)
        else:
            embedding = self.embeddings(masked_indices[:, :, 0])

            for idx in range(1, len(self._masks)):
                embedding += self.embeddings(masked_indices[:, :, idx])

            embedding /= len(self._masks)

        return embedding
