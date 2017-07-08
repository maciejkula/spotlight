.. image:: /docs/_static/img/spotlight.png

---------------------------------------------------------------------

.. inclusion-marker-do-not-remove

.. image:: https://travis-ci.org/maciejkula/spotlight.svg?branch=master
   :target: https://travis-ci.org/maciejkula/spotlight

|

Spotlight uses `PyTorch <http://pytorch.org/>`_ to build both deep and shallow
recommender models. By providing both a slew of building blocks for loss functions
(various pointwise and pairwise ranking losses), representations (shallow
factorization representations, deep sequence models), and utilities for fetching
(or generating) recommendation datasets, it aims to be a tool for rapid exploration
and prototyping of new recommender models.

Installation
------------
.. code-block:: python

   conda install -c maciejkula -c soumith spotlight=0.1.0


Usage
-----

.. testcode::

    from spotlight.cross_validation import random_train_test_split
    from spotlight.datasets.movielens import get_movielens_dataset
    from spotlight.evaluation import rmse_score
    from spotlight.factorization.explicit import ExplicitFactorizationModel

    dataset = get_movielens_dataset(variant='100K')

    train, test = random_train_test_split(dataset)

    model = ExplicitFactorizationModel(n_iter=1)
    model.fit(train)

    rmse = rmse_score(model, test)

.. testoutput::
   :hidden:
