
# Causal convolutions
Recurrent neural networks are not the only way of effectively representing sequences: convolutions can also do the job. In particular, we can use _causal convolutions_: convolutional filters applied to the sequence in a left-to-right fashion, emitting a representation at each step. They are _causal_ in that the their output at time $t$ is conditional on input up to $t-1$: this is necessary to ensure that they do not have access to the elements of the sequence we are trying to predict. 


Like LSTMs, causal convolutions can model sequences with long-term dependencies. This is achieved in two ways: stacking convolutional layers (with padding, every convolutional layer preserves the shape of the input), and _dilation_: insertion of gaps into the convolutional filters (otherwise known as _atrous_ convolutions).

The [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) paper uses dilated causal convolutions to model audio:

---

<img src="https://storage.googleapis.com/deepmind-live-cms-alt/documents/BlogPost-Fig2-Anim-160908-r01.gif" alt="Causal convolutions" style="width: 600x;"/>

---

Using convolutional rather than recurrent networks for representing sequences has a couple of advantages, as described in [this](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f) blog post: 

1. Parallelization: RNNs needs to process inputs in a sequential fashion, one time-step at a time. In contrast, a CNN can perform convolutions across the entire sequence in parallel.
2. Convolutional representations are less likely to be bottlenecked by the fixed size of the RNN representation, or by the distance between the hidden output and the input in long sequences. Using convolutional networks, the distance between the output and is determined by the depth of the network, and is independent of the length of the sequence (see section 1 of [Neural Machine Translation in Linear Time](https://arxiv.org/pdf/1610.10099.pdf)).

## Causal convolutions in Spotlight
Spotlight implements causal convolution models as part of its [sequence models](https://maciejkula.github.io/spotlight/sequence/sequence.html) package, alongside more traditional recurrent and pooling models. The implementation uses:

1. embedding layers for input,
1. stackedn CNNs using tanh or relu non-linearities,
2. residual connections between all layers, and
3. configurable kernel size and dilations.

## Experiments
To see how causal CNNs compare to more traditional sequence models we can have a look at how they perform at predicting the next rated movie on the Movielens 1M dataset. With 1 million interactions spread among 6000 users and around 4000 movies it should be small enough to run quick experiments, but large enough to yield meaningful results.

I chose to split the dataset into 80% train, and 10% test and validation sets. I construct 200-long sequences by splitting each user's item sequence into 200-long chunks; if a chunk is shorter than 200 elements, it's padded with zeros.


```python
import numpy as np

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import user_based_train_test_split

max_sequence_length = 200
min_sequence_length = 20
step_size = 200

random_state = np.random.RandomState(100)

# Get data
dataset = get_movielens_dataset('1M')

# Split into train, test, validation
train, rest = user_based_train_test_split(dataset,
                                          random_state=random_state)
test, validation = user_based_train_test_split(rest,
                                               test_percentage=0.5,
                                               random_state=random_state)

# Convert to sequences
train = train.to_sequence(max_sequence_length=max_sequence_length,
                          min_sequence_length=min_sequence_length,
                          step_size=step_size)
test = test.to_sequence(max_sequence_length=max_sequence_length,
                        min_sequence_length=min_sequence_length,
                        step_size=step_size)
validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                    min_sequence_length=min_sequence_length,
                                    step_size=step_size)
```

To choose hyperparameterss, I run a quick, coarse grained hyperparameter search, using random sampling to draw 100 hyperparameter sets:


```python
from sklearn.model_selection import ParameterSampler

import torch

from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score

CUDA = torch.cuda.is_available()

LEARNING_RATES = [1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'hinge', 'adaptive_hinge', 'pointwise']
BATCH_SIZE = [8, 16, 32, 256]
EMBEDDING_DIM = [8, 16, 32, 64, 128, 256]
N_ITER = list(range(5, 20))
L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]

def sample_cnn_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
        'kernel_width': [3, 5, 7],
        'num_layers': list(range(1, 10)),
        'dilation_multiplier': [1, 2],
        'nonlinearity': ['tanh', 'relu'],
        'residual': [True, False]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        params['dilation'] = list(params['dilation_multiplier'] ** (i % 8)
                                  for i in range(params['num_layers']))

        yield params
        
def evaluate_cnn_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    net = CNNNet(train.num_items,
                 embedding_dim=h['embedding_dim'],
                 kernel_width=h['kernel_width'],
                 dilation=h['dilation'],
                 num_layers=h['num_layers'],
                 nonlinearity=h['nonlinearity'],
                 residual_connections=h['residual'])

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation=net,
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)
    val_mrr = sequence_mrr_score(model, validation)

    return test_mrr, val_mrr
```

Running a single iteration should work roughly as follows:


```python
hyperparams = list(sample_cnn_hyperparameters(np.random.RandomState(), 1))[0]

# Fitting and validating the model may take a couple of minutes if you don't have a GPU:
# run at your own risk!
# test_mrr, validation_mrr = evaluate_cnn_model(hyperparams, train, test, validation, None)
```

## Results

The results are as follows:

---

<center> __Causal convolution results__ </center>

validation_mrr|test_mrr|residual|nonlinearity|     loss     |num_layers|kernel_width|         dilation          |embedding_dim
-------------:|-------:|--------|------------|--------------|---------:|-----------:|---------------------------|------------:
       0.07221| 0.07951|True    |relu        |adaptive_hinge|         3|           3|[1, 2, 4]            |          256
       0.06583| 0.06624|True    |relu        |adaptive_hinge|         5|           5|[1, 2, 4, 8, 16]|           32
       0.06563| 0.07177|True    |relu        |adaptive_hinge|         5|           5|[1, 1, 1, 1, 1]  |          128
       0.05832| 0.06827|True    |relu        |hinge         |         4|           5|[1, 1, 1, 1]       |          128
       0.05771| 0.04971|True    |tanh        |hinge         |         9|           7|[1, 1, 1, 1, 1, 1, 1, 1, 1]|           64

---

It's difficult to draw clear-cut conclusions about the effect of each hyperparameter, but it looks like:

- The model works, making predictions substantially better than random.
- The ReLU nonlinearity and the adaptive hinge losses work best.
- More than one CNN layer is necessary to achieve good results.

To compare causal convolutions with more traditional sequence models I run similar hyperparameter searches for [LSTM-based representations](https://maciejkula.github.io/spotlight/sequence/representations.html#spotlight.sequence.representations.LSTMNet) and [pooling representations](https://maciejkula.github.io/spotlight/sequence/representations.html#spotlight.sequence.representations.PoolNet). The pooling representation is a simple averaging of item embedding across the sequence; the LSTM-based model runs an LSTM along a user's interactions, using the hidden state for prediction of the next element at each step. The results are as follows:

---

<center> __LSTM results__ </center>

validation_mrr|test_mrr|batch_size|embedding_dim|  l2   |learning_rate|     loss     |n_iter
-------------:|-------:|---------:|------------:|------:|------------:|--------------|-----:
       0.08291| 0.07637|        16|           64| 0.0000|         0.01|adaptive_hinge|    15
       0.07811| 0.08081|       256|           32| 0.0000|         0.05|adaptive_hinge|    11
       0.07690| 0.07910|        32|           16| 0.0000|         0.01|adaptive_hinge|    13
       0.07569| 0.07081|        16|           64| 0.0000|         0.01|adaptive_hinge|    12
       0.07349| 0.07534|       256|            8| 0.0000|         0.01|adaptive_hinge|    10

---

---

<center> __Pooling results__ </center>

validation_mrr|test_mrr|batch_size|embedding_dim|  l2  |learning_rate|     loss     |n_iter
-------------:|-------:|---------:|------------:|-----:|------------:|--------------|-----:
       0.01785| 0.01339|        16|          256|  0.00|         0.10|adaptive_hinge|    19
       0.01720| 0.01346|        32|            8|  0.00|         0.05|hinge         |    14
       0.01504| 0.01459|        16|           64|  0.00|         0.01|adaptive_hinge|    15
       0.01455| 0.01632|       256|            8|  0.00|         0.10|hinge         |     7
       0.01421| 0.01541|       256|           32|  0.00|         0.05|adaptive_hinge|    11

---

A single layer LSTM seems to outperform causal convolutions, by an over 10% margin, helped by the [adaptive hinge](https://maciejkula.github.io/spotlight/losses.html#spotlight.losses.adaptive_hinge_loss) loss. Simple pooling performs quite badly.

### Avenues to explore

There are a couple of avenues to explore.

- gated CNN units
- stacking LSTMs, dropout
- deep averaging networks


```python

```
