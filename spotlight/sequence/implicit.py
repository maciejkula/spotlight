import numpy as np

import torch

import torch.optim as optim

from torch.autograd import Variable


from spotlight.losses import (bpr_loss,
                              hinge_loss,
                              pointwise_loss)
from spotlight.sequence.representations import PADDING_IDX, CNNNet, LSTMNet, PoolNet
from spotlight.sampling import sample_items
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle


class ImplicitSequenceModel(object):

    def __init__(self,
                 loss='pointwise',
                 representation='pooling',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer=None,
                 use_cuda=False,
                 sparse=False,
                 random_state=None):

        assert loss in ('pointwise',
                        'bpr',
                        'hinge',
                        'adaptive_hinge')

        if isinstance(representation, str):
            assert representation in ('pooling',
                                      'cnn',
                                      'lstm')

        self._loss = loss
        self._representation = representation
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._optimizer = None
        self._random_state = random_state or np.random.RandomState()

        self._num_items = None
        self._net = None

        set_seed(self._random_state.randint(-10**8, 10**8),
                 cuda=self._use_cuda)

    def fit(self, interactions, verbose=False):
        """
        """

        sequences = interactions.sequences.astype(np.int64)

        self._num_items = interactions.num_items

        if self._representation == 'pooling':
            self._net = PoolNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        elif self._representation == 'cnn':
            self._net = CNNNet(self._num_items,
                               self._embedding_dim,
                               sparse=self._sparse)
        elif self._representation == 'lstm':
            self._net = LSTMNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        else:
            self._net = self._representation

        if self._optimizer is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )

        if self._loss == 'pointwise':
            loss_fnc = pointwise_loss
        elif self._loss == 'bpr':
            loss_fnc = bpr_loss
        else:
            loss_fnc = hinge_loss

        for epoch_num in range(self._n_iter):

            sequences = shuffle(sequences,
                                random_state=self._random_state)

            sequences_tensor = gpu(torch.from_numpy(sequences),
                                   self._use_cuda)

            epoch_loss = 0.0

            for batch_sequence in minibatch(sequences_tensor,
                                            batch_size=self._batch_size):

                sequence_var = Variable(batch_sequence)

                user_representation, _ = self._net.user_representation(
                    sequence_var
                )

                positive_prediction = self._net(user_representation,
                                                sequence_var)

                if self._loss == 'adaptive_hinge':
                    raise NotImplementedError
                else:
                    negative_items = sample_items(
                        self._num_items,
                        batch_sequence.size(),
                        random_state=self._random_state)
                    negative_var = Variable(
                        gpu(torch.from_numpy(negative_items))
                    )
                    negative_prediction = self._net(user_representation,
                                                    negative_var)

                self._optimizer.zero_grad()

                loss = loss_fnc(positive_prediction,
                                negative_prediction,
                                mask=(sequence_var != PADDING_IDX))
                epoch_loss += loss.data[0]

                loss.backward()
                self._optimizer.step()

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    # def _get_adaptive_negatives(self, user_ids, num_neg_candidates=5):

    #     negatives = Variable(
    #         gpu(
    #             torch.from_numpy(
    #                 self._random_state
    #                 .randint(0, self._num_items,
    #                          (len(user_ids), num_neg_candidates))),
    #             self._use_cuda)
    #     )
    #     negative_predictions = self._net(
    #         user_ids.repeat(num_neg_candidates, 1).transpose(0, 1),
    #         negatives
    #     ).view(-1, num_neg_candidates)

    #     best_negative_prediction, _ = negative_predictions.max(1)

    #     return best_negative_prediction

    def predict(self, sequences, item_ids=None):
        """
        """

        sequences = np.atleast_2d(sequences)

        if item_ids is None:
            item_ids = np.arange(self._num_items).reshape(-1, 1)

        sequences = torch.from_numpy(sequences.astype(np.int64).reshape(1, -1))
        item_ids = torch.from_numpy(item_ids.astype(np.int64))

        sequence_var = Variable(gpu(sequences, self._use_cuda))
        item_var = Variable(gpu(item_ids, self._use_cuda))

        _, sequence_representations = self._net.user_representation(sequence_var)
        out = self._net(sequence_representations.repeat(len(item_var), 1),
                        item_var)

        return cpu(out.data).numpy().flatten()
