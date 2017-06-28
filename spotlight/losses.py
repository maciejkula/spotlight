import torch

import torch.nn.functional as F

from spotlight.torch_utils import assert_no_grad


def pointwise_loss(positive_predictions, negative_predictions):

    positives_loss = (1.0 - F.sigmoid(positive_predictions))
    negatives_loss = F.sigmoid(negative_predictions)

    return torch.cat([positives_loss, negatives_loss]).mean()


def bpr_loss(positive_predictions, negative_predictions):

    return (1.0 - F.sigmoid(positive_predictions -
                            negative_predictions)).mean()


def hinge_loss(positive_predictions, negative_predictions):

    return torch.mean(torch.clamp(negative_predictions -
                                  positive_predictions +
                                  1.0, 0.0))


def truncated_regression_loss(observed_rating,
                              positive_observation_probability,
                              positive_predicted_rating,
                              predicted_rating_stddev,
                              negative_predicted_probability):

    assert_no_grad(observed_rating)

    positives_likelihood = (torch.log(positive_observation_probability) -
                            0.5 * torch.log(predicted_rating_stddev ** 2) -
                            (0.5 * (positive_predicted_rating -
                                    observed_rating) ** 2 /
                             (predicted_rating_stddev ** 2)))
    negatives_likelihood = torch.log(1.0 - negative_predicted_probability)

    return torch.cat([-positives_likelihood, -negatives_likelihood]).mean()


def regression_loss(observed_ratings,
                    predicted_ratings):

    assert_no_grad(observed_ratings)

    return ((observed_ratings - predicted_ratings) ** 2).mean()


def poisson_loss(observed_ratings,
                 predicted_ratings):

    assert_no_grad(observed_ratings)

    return (predicted_ratings - observed_ratings * torch.log(predicted_ratings)).mean()
