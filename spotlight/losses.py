"""
Loss functions for recommender models.

The pointwise, BPR, and hinge losses are a good fit for
implicit feedback models trained through negative sampling.

The regression and Poisson losses are used for explicit feedback
models.
"""

import torch

import torch.nn.functional as F

from spotlight.torch_utils import assert_no_grad


def pointwise_loss(positive_predictions, negative_predictions, mask=None):
    """
    Logistic loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    positives_loss = (1.0 - F.sigmoid(positive_predictions))
    negatives_loss = F.sigmoid(negative_predictions)

    loss = (positives_loss + negatives_loss)

    if mask is not None:
        loss = loss * mask.float()

    return loss.mean()


def bpr_loss(positive_predictions, negative_predictions, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    loss = (1.0 - F.sigmoid(positive_predictions -
                            negative_predictions))

    if mask is not None:
        loss = loss * mask.float()

    return loss.mean()


def hinge_loss(positive_predictions, negative_predictions):
    """
    Hinge pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    return torch.mean(torch.clamp(negative_predictions -
                                  positive_predictions +
                                  1.0, 0.0))


def regression_loss(observed_ratings, predicted_ratings):
    """
    Regression loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    negative_predictions: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    return ((observed_ratings - predicted_ratings) ** 2).mean()


def poisson_loss(observed_ratings, predicted_ratings):
    """
    Poisson loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    negative_predictions: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    return (predicted_ratings - observed_ratings * torch.log(predicted_ratings)).mean()
