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


def _weighted_loss(loss, sample_weights=None, mask=None):
    """Sample weight and mask handler for loss functions.
    If both sample_weights and mask are specified, sample_weights will override
    as one may zero-out, as well as scale, certain entries via the weights.

    Parameters
    ----------

    loss: tensor
        Tensor with element-wise losses from one of the loss functions in this
        file.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    if sample_weights is not None:
        loss = loss * sample_weights
        return loss.sum() / sample_weights.sum()

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def pointwise_loss(
        positive_predictions, negative_predictions,
        sample_weights=None, mask=None):
    """
    Logistic loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
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

    return _weighted_loss(loss, sample_weights, mask)


def bpr_loss(
        positive_predictions, negative_predictions,
        sample_weights=None, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """

    loss = (1.0 - F.sigmoid(positive_predictions -
                            negative_predictions))

    return _weighted_loss(loss, sample_weights, mask)


def hinge_loss(
        positive_predictions, negative_predictions,
        sample_weights=None, mask=None):
    """
    Hinge pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       1.0, 0.0)

    return _weighted_loss(loss, sample_weights, mask)


def adaptive_hinge_loss(
        positive_predictions, negative_predictions,
        sample_weights=None, mask=None):
    """
    Adaptive hinge pairwise loss function. Takes a set of predictions
    for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closest to violating the
    ranking implicit in the pattern of user interactions.

    Approximates the idea of Weighted Approximate-Rank Pairwise (WARP) loss
    introduced in [2]_

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Iterable of tensors containing predictions for sampled negative items.
        More tensors increase the likelihood of finding ranking-violating
        pairs, but risk overfitting.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
       Scaling up to large vocabulary image annotation." IJCAI.
       Vol. 11. 2011.
    """

    highest_negative_predictions, _ = torch.max(negative_predictions, 0)

    return hinge_loss(
        positive_predictions,
        highest_negative_predictions.squeeze(),
        sample_weights=sample_weights,
        mask=mask
    )


def regression_loss(
        observed_ratings, predicted_ratings,
        sample_weights=None, mask=None):
    """
    Regression loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)
    loss = (observed_ratings - predicted_ratings) ** 2

    return _weighted_loss(loss, sample_weights, mask)


def poisson_loss(
        observed_ratings, predicted_ratings,
        sample_weights=None, mask=None):
    """
    Poisson loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)
    loss = predicted_ratings - observed_ratings * torch.log(predicted_ratings)

    return _weighted_loss(loss, sample_weights, mask)


def logistic_loss(
        observed_ratings, predicted_ratings,
        sample_weights=None, mask=None):
    """
    Logistic loss for explicit data.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings which
        should be +1 or -1 for this loss function.
    predicted_ratings: tensor
        Tensor containing rating predictions.
    sample_weights: tensor, optional
        Tensor containing weights to scale the loss by.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    assert_no_grad(observed_ratings)

    # Convert target classes from (-1, 1) to (0, 1)
    observed_ratings = torch.clamp(observed_ratings, 0, 1)

    if sample_weights is not None or mask is not None:
        loss = F.binary_cross_entropy_with_logits(
            predicted_ratings,
            observed_ratings,
            size_average=False
        )
        return _weighted_loss(loss, sample_weights, mask)

    return F.binary_cross_entropy_with_logits(
        predicted_ratings,
        observed_ratings,
        size_average=True
    )
