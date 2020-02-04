Causal convolutions for sequence-based recommendations
======================================================

Using sequences of user-item interactions as an input for recommender
models has a number of attractive properties. Firstly, it recognizes
that recommending the next item that a user may want to buy or see is
precisely the goal we are trying to achieve. Secondly, it's plausible
that the ordering of users' interactions carries additional information
over and above just the identities of items they have interacted with.
For example, a user is more likely to watch the next episode of a given
TV series if they've just finished the previous episode. Finally, when
the sequence of past interactions rather than the identity of the user
is the input to a model, online systems can incorporate new users (and
old users' new actions) in real time. They are fed to the existing
model, and do not require a new model to be fit to incorporate new
information (unlike factorization models).

Recurrent neural networks are the most natural way of modelling such
sequence problems. In recommendations, gated recurrent units (GRUs) have
been used with success in the `Session-based recommendations with
recurrent neural networks <https://arxiv.org/abs/1511.06939>`__ paper.
Spotlight implements a similar model using `LSTM
units <https://maciejkula.github.io/spotlight/sequence/representations.html#spotlight.sequence.representations.LSTMNet>`__
as one of its sequence representations.

Causal convolutions
-------------------

But recurrent neural networks are not the only way of effectively
representing sequences: convolutions can also do the job. In particular,
we can use *causal convolutions*: convolutional filters applied to the
sequence in a left-to-right fashion, emitting a representation at each
step. They are *causal* in that the their output at time :math:`t` is
conditional on input up to :math:`t-1`: this is necessary to ensure that
they do not have access to the elements of the sequence we are trying to
predict.

Like LSTMs, causal convolutions can model sequences with long-term
dependencies. This is achieved in two ways: stacking convolutional
layers (with padding, every convolutional layer preserves the shape of
the input), and *dilation*: insertion of gaps into the convolutional
filters (otherwise known as *atrous* convolutions).

Causal atrous convolutions
**************************

.. figure:: https://storage.googleapis.com/deepmind-live-cms-alt/documents/BlogPost-Fig2-Anim-160908-r01.gif
   :target: https://travis-ci.org/maciejkula/spotlight
   :align: center

|

Causal convolutions have been used in several recent high-profile
papers:

-  the `WaveNet <https://arxiv.org/pdf/1609.03499.pdf>`__ paper uses
   dilated causal convolutions to model audio,
-  the
   `PixelCNN <http://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders>`_
   paper uses them to generate images, and
-  the `Neural Machine Translation in Linear
   Time <https://arxiv.org/abs/1610.10099>`_ paper uses them for
   machine translation.

Using convolutional rather than recurrent networks for representing
sequences has a couple of advantages, as described in
`this <https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f>`_
blog post:

1. Parallelization: RNNs needs to process inputs in a sequential
   fashion, one time-step at a time. In contrast, a CNN can perform
   convolutions across the entire sequence in parallel.
2. Convolutional representations are less likely to be bottlenecked by
   the fixed size of the RNN representation, or by the distance between
   the hidden output and the input in long sequences. Using
   convolutional networks, the distance between the output and is
   determined by the depth of the network, and is independent of the
   length of the sequence (see section 1 of `Neural Machine Translation
   in Linear Time <https://arxiv.org/pdf/1610.10099.pdf>`_).

Causal convolutions in Spotlight
--------------------------------

Spotlight implements causal convolution models as part of its `sequence
models <https://maciejkula.github.io/spotlight/sequence/sequence.html>`_
package, alongside more traditional recurrent and pooling models. The
`Spotlight
implementation <https://maciejkula.github.io/spotlight/sequence/representations.html#spotlight.sequence.representations.CNNNet>`_
has the following characteristics:

1. Embedding layers for input and output. The weights of the input and
   output embedding layers are tied: the representation used by an item
   when encoding the sequence is the same as the one used in prediction.
2. Stacked CNNs using tanh or relu non-linearities. The sequence is
   appropriately padded to ensure that future elements of the sequence
   are never in the receptive field of the network at a given time.
3. Residual connections can be applied between all layers.
4. Kernel size and dilation can be specified separately for each stacked
   convolutional layer.

The model is trained using one of Spotlight's implicit feedback
`losses <https://maciejkula.github.io/spotlight/losses.html>`_,
including pointwise (logistic and hinge) and pairwise (BPR as well as
WARP-like adaptive hinge) losses. As with other Spotlight sequence
models, the loss is computed for all the time steps of the sequence in
one pass: for all timesteps :math:`t` in the sequence, a prediction
using elements up to :math:`t-1` is made, and the loss is averaged along
both the time and the minibatch axis. This leads to significant training
speed-ups relative to only computing the loss for the last element in
the sequence.

Experiments
-----------

To see how causal CNNs compare to more traditional sequence models we
can have a look at how they perform at predicting the next rated movie
on the `Movielens 1M dataset <https://grouplens.org/datasets/movielens/1m/>`_. With 1 million interactions spread among
6000 users and around 4000 movies it should be small enough to run quick
experiments, but large enough to yield meaningful results.

I chose to split the dataset into 80% train, and 10% test and validation
sets. I construct 200-long sequences by splitting each user's item
sequence into 200-long chunks; if a chunk is shorter than 200 elements,
it's padded with zeros. I use `mean reciprocal rank <https://en.wikipedia.org/wiki/Mean_reciprocal_rank>`_ (MRR) as the evaluation
metric.

To choose hyperparameters, I run a quick, coarse grained hyperparameter
search, using random sampling to draw 100 hyperparameter sets. With the
data and hyperparameters ready, fitting and evaluating the model is
relatively simple:

.. code:: python

    import torch

    from spotlight.sequence.implicit import ImplicitSequenceModel
    from spotlight.sequence.representations import CNNNet
    from spotlight.evaluation import sequence_mrr_score

            
    net = CNNNet(train.num_items,
                 embedding_dim=hyperparameters['embedding_dim'],
                 kernel_width=hyperparameters['kernel_width'],
                 dilation=hyperparameters['dilation'],
                 num_layers=hyperparameters['num_layers'],
                 nonlinearity=hyperparameters['nonlinearity'],
                 residual_connections=hyperparameters['residual'])

    model = ImplicitSequenceModel(loss=hyperparameters['loss'],
                                  representation=net,
                                  batch_size=hyperparameters['batch_size'],
                                  learning_rate=hyperparameters['learning_rate'],
                                  l2=hyperparameters['l2'],
                                  n_iter=hyperparameters['n_iter'],
                                  use_cuda=torch.cuda.is_available(),
                                  random_state=random_state)

    model.fit(train)

    test_mrr = sequence_mrr_score(model, test)
    val_mrr = sequence_mrr_score(model, validation)

Fitting the models is fairly quick, taking at most two or three minutes on a single K80 GPU. The code for the experiments is available in the
experiments folder of the Spotlight repo.

Results
-------

The results are as follows:

Causal convolution results
**************************

+-------------------+-------------+------------+----------------+-------------------+---------------+-----------------+-------------------------------+------------------+
| validation\_mrr   | test\_mrr   | residual   | nonlinearity   | loss              | num\_layers   | kernel\_width   | dilation                      | embedding\_dim   |
+===================+=============+============+================+===================+===============+=================+===============================+==================+
| 0.0722109         | 0.0795061   | True       | relu           | adaptive\_hinge   | 3             | 3               | [1, 2, 4]                     | 256              |
+-------------------+-------------+------------+----------------+-------------------+---------------+-----------------+-------------------------------+------------------+
| 0.0658315         | 0.0662418   | True       | relu           | adaptive\_hinge   | 5             | 5               | [1, 2, 4, 8, 16]              | 32               |
+-------------------+-------------+------------+----------------+-------------------+---------------+-----------------+-------------------------------+------------------+
| 0.0656252         | 0.0717681   | True       | relu           | adaptive\_hinge   | 5             | 5               | [1, 1, 1, 1, 1]               | 128              |
+-------------------+-------------+------------+----------------+-------------------+---------------+-----------------+-------------------------------+------------------+
| 0.0583223         | 0.0682682   | True       | relu           | hinge             | 4             | 5               | [1, 1, 1, 1]                  | 128              |
+-------------------+-------------+------------+----------------+-------------------+---------------+-----------------+-------------------------------+------------------+
| 0.0577055         | 0.0497131   | True       | tanh           | hinge             | 9             | 7               | [1, 1, 1, 1, 1, 1, 1, 1, 1]   | 64               |
+-------------------+-------------+------------+----------------+-------------------+---------------+-----------------+-------------------------------+------------------+

It's difficult to draw clear-cut conclusions about the effect of each
hyperparameter, but it looks like:

-  The model works, making predictions substantially better than random.
-  The ReLU nonlinearity and the adaptive hinge losses work best.
-  More than one CNN layer is necessary to achieve good results.

To compare causal convolutions with more traditional sequence models I
run similar hyperparameter searches for `LSTM-based
representations <https://maciejkula.github.io/spotlight/sequence/representations.html#spotlight.sequence.representations.LSTMNet>`_
and `pooling
representations <https://maciejkula.github.io/spotlight/sequence/representations.html#spotlight.sequence.representations.PoolNet>`_.
The pooling representation is a simple averaging of item embedding across the sequence; the LSTM-based model runs an LSTM along a user's
interactions, using the hidden state for prediction of the next element at each step. The results are as follows:

LSTM results
************

+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| validation\_mrr   | test\_mrr   | batch\_size   | embedding\_dim   | l2      | learning\_rate   | loss              | n\_iter   |
+===================+=============+===============+==================+=========+==================+===================+===========+
| 0.082913          | 0.0763708   | 16            | 64               | 0       | 0.01             | adaptive\_hinge   | 15        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.078108          | 0.0808093   | 256           | 32               | 0       | 0.05             | adaptive\_hinge   | 11        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.0769014         | 0.0791023   | 32            | 16               | 1e-06   | 0.01             | adaptive\_hinge   | 13        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.0756949         | 0.0708071   | 16            | 64               | 1e-05   | 0.01             | adaptive\_hinge   | 12        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.0734895         | 0.0753369   | 256           | 8                | 1e-05   | 0.01             | adaptive\_hinge   | 10        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| validation\_mrr   | test\_mrr   | batch\_size   | embedding\_dim   | l2      | learning\_rate   | loss              | n\_iter   |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+

Pooling results
***************

+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| validation\_mrr   | test\_mrr   | batch\_size   | embedding\_dim   | l2      | learning\_rate   | loss              | n\_iter   |
+===================+=============+===============+==================+=========+==================+===================+===========+
| 0.0178542         | 0.0133928   | 16            | 256              | 1e-05   | 0.1              | adaptive\_hinge   | 19        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.0172026         | 0.0134581   | 32            | 8                | 0       | 0.05             | hinge             | 14        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.0150402         | 0.0145902   | 16            | 64               | 0       | 0.01             | adaptive\_hinge   | 15        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.0145492         | 0.0163207   | 256           | 8                | 0       | 0.1              | hinge             | 7         |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+
| 0.0142107         | 0.0154118   | 256           | 32               | 0       | 0.05             | adaptive\_hinge   | 11        |
+-------------------+-------------+---------------+------------------+---------+------------------+-------------------+-----------+

A single layer LSTM seems to outperform causal convolutions, by an over 10% margin, helped by the `adaptive
hinge <https://maciejkula.github.io/spotlight/losses.html#spotlight.losses.adaptive_hinge_loss>`_ loss. Simple pooling performs quite badly.

Avenues to explore
------------------

It looks like causal convolutions need some more work before beating
recurrent networks. There are a couple of possible avenues for making
them better:

1. Gated CNN units. The WaveNet paper uses gated CNN units. These
   consist of two convolutional layers: one using the tanh and the other
   (the gate) using the sigmoid nonlinearity. They are then multiplied
   together to achieve the sort of gating effect more commonly seen in
   recurrent networks. I have run some small scale experiements using
   gated CNN units, but I haven't managed to extract meaningful accuracy
   gains from them.
2. Batch normalization. Batch normalization is key to training many
   multi-layer convolutional networks; maybe it would be of use here?
   Again, my experiments failed to show a benefit, but I may have missed
   a small but crucial trick of the trade.
3. Skip connections. Would skip connections - in addition to residual
   connections - help with the accuracy?

I'd love to get some input on these. If you have suggestions, let me
know on `Twitter <https://twitter.com/Maciej_Kula>`_ or open an
issue or PR in `Spotlight <https://github.com/maciejkula/spotlight>`_.

