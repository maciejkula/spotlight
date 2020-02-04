
# Explicit feedback movie recommendations
In this example, we'll build a quick explicit feedback recommender system: that is, a model that takes into account explicit feedback signals (like ratings) to recommend new content.

We'll use an approach first made popular by the [Netflix prize](http://www.netflixprize.com/) contest: [matrix factorization](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf). 

The basic idea is very simple:

1. Start with user-item-rating triplets, conveying the information that user _i_ gave some item _j_ rating _r_.
2. Represent both users and items as high-dimensional vectors of numbers. For example, a user could be represented by `[0.3, -1.2, 0.5]` and an item by `[1.0, -0.3, -0.6]`.
3. The representations should be chosen so that, when we multiplied together (via [dot products](https://en.wikipedia.org/wiki/Dot_product)), we can recover the original ratings.
4. The utility of the model then is derived from the fact that if we multiply the user vector of a user with the item vector of some item they _have not_ rated, we hope to obtain a prediction for the rating they would have given to it had they seen it.

<img src="static/matrix_factorization.png" alt="Matrix factorization" style="width: 600px;"/>

Spotlight fits models such as these using [stochastic gradient descent](http://cs231n.github.io/optimization-1/). The procedure goes roughly as follows:

1. Start with representing users and items by randomly chosen vectors. Because they are random, they are not going to give useful recommendations, but we are going to improve them as we fit the model.
2. Go through the (user, item, rating) triplets in the dataset. For every triplet, compute the rating that the model predicts by multiplying the user and item vectors together, and compare the result with the actual rating: the closer they are, the better the model.
3. If the predicted rating is too low, adjust the user and item vectors (by a small amount) to increase the prediction.
4. If the predicted rating is too high, adjust the vectors to decrease it.
5. Continue iterating over the training triplets until the model's accuracy stabilizes.

## The data



We start with importing a famous dataset, the [Movielens 100k dataset](https://grouplens.org/datasets/movielens/100k/). It contains 100,000 ratings (between 1 and 5) given to 1683 movies by 944 users:


```python
import numpy as np

from spotlight.datasets.movielens import get_movielens_dataset

dataset = get_movielens_dataset(variant='100K')
print(dataset)
```

    <Interactions dataset (944 users x 1683 items x 100000 interactions)>


The `dataset` object is an instance of an `Interactions` [class](https://maciejkula.github.io/spotlight/interactions.html#spotlight.interactions.Interactions), a fairly light-weight wrapper that Spotlight users to hold the arrays that contain information about an interactions dataset (such as user and item ids, ratings, and timestamps).

## The model

We can feed our dataset to the [`ExplicitFactorizationModel`](https://maciejkula.github.io/spotlight/factorization/explicit.html#spotlight.factorization.explicit.ExplicitFactorizationModel) class - and sklearn-like object that allows us to train and evaluate the explicit factorization models.

Internally, the model uses the [`BilinearNet`](https://maciejkula.github.io/spotlight/factorization/representations.html#spotlight.factorization.representations.BilinearNet) class to represents users and items. It's composed of a 4 [embedding layers](http://pytorch.org/docs/master/nn.html?highlight=embedding#torch.nn.Embedding):

- a `(num_users x latent_dim)` embedding layer to represent users,
- a `(num_items x latent_dim)` embedding layer to represent items,
- a `(num_users x 1)` embedding layer to represent user biases, and
- a `(num_items x 1)` embedding layer to represent item biases.

Together, these give us the predictions. Their accuracy is evaluated using one of the Spotlight [losses](https://maciejkula.github.io/spotlight/losses.html). In this case, we'll use the [regression loss](https://maciejkula.github.io/spotlight/losses.html#spotlight.losses.regression_loss), which is simply the squared difference between the true and the predicted rating.


```python
import torch

from spotlight.factorization.explicit import ExplicitFactorizationModel

model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=128,  # latent dimensionality
                                   n_iter=10,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   l2=1e-9,  # strength of L2 regularization
                                   learning_rate=1e-3,
                                   use_cuda=torch.cuda.is_available())
```

In order to fit and evaluate the model, we need to split it into a train and a test set:


```python
from spotlight.cross_validation import random_train_test_split

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

print('Split into \n {} and \n {}.'.format(train, test))
```

    Split into 
     <Interactions dataset (944 users x 1683 items x 80000 interactions)> and 
     <Interactions dataset (944 users x 1683 items x 20000 interactions)>.


With the data ready, we can go ahead and fit the model. This should take less than a minute on the CPU, and we should see the loss decreasing as the model is learning better and better representations for the user and items in our dataset.


```python
model.fit(train, verbose=True)
```

    Epoch 0: loss 13.11806030514874
    Epoch 1: loss 7.320562576945824
    Epoch 2: loss 1.75225291825548
    Epoch 3: loss 1.0712461079223246
    Epoch 4: loss 0.943024439147756
    Epoch 5: loss 0.8985876848426047
    Epoch 6: loss 0.878033770790583
    Epoch 7: loss 0.8649711623976503
    Epoch 8: loss 0.8586649985253056
    Epoch 9: loss 0.846391685401337


Now that the model is estimated, how good are its predictions?


```python
from spotlight.evaluation import rmse_score

train_rmse = rmse_score(model, train)
test_rmse = rmse_score(model, test)

print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))
```

    Train RMSE 0.907, test RMSE 0.946


## Conclusions

This is a fairly simple model, and can be extended by adding side-information, adding more non-linear layers, and so on.

However, before plunging into such extensions, it is worth knowing that models using explicit ratings have fallen out of favour both in [academia](https://pdfs.semanticscholar.org/8e8e/cc4591f6d919f6ad247e7ef3300de2fed7a3.pdf)  and in [industry](https://media.netflix.com/en/company-blog/goodbye-stars-hello-thumbs). It is now widely accepted that _what_ people choose to interact with is more meaningful than how they rate the interactions they have.

These scenarios are called _implicit feedback_ settings. If you're interested in building these models, have a look at Spotlight's [implicit factorization](https://maciejkula.github.io/spotlight/factorization/implicit.html) models, as well as the [implicit sequence models](https://maciejkula.github.io/spotlight/sequence/representations.html) which aim to explicitly model the sequential nature of interaction data.
