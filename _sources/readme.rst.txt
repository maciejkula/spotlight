.. image:: _static/img/spotlight.png

---------------------------------------------------------------------

.. inclusion-marker-do-not-remove

.. image:: https://travis-ci.org/maciejkula/spotlight.svg?branch=master
   :target: https://travis-ci.org/maciejkula/spotlight

.. image:: https://ci.appveyor.com/api/projects/status/jq5e76a7a08ra2ji/branch/master?svg=true
   :target: https://ci.appveyor.com/project/maciejkula/spotlight/branch/master

.. image:: https://badges.gitter.im/gitterHQ/gitter.png
   :target: https://gitter.im/spotlight-recommendations/Lobby

.. image:: https://anaconda.org/maciejkula/spotlight/badges/version.svg
   :target: https://anaconda.org/maciejkula/spotlight

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://maciejkula.github.io/spotlight/

.. image:: https://img.shields.io/badge/progress%20tracker-trello-brightgreen.svg
   :target: https://trello.com/b/G5iFgS1W/spotlight

|

Spotlight uses `PyTorch <http://pytorch.org/>`_ to build both deep and shallow
recommender models. By providing both a slew of building blocks for loss functions
(various pointwise and pairwise ranking losses), representations (shallow
factorization representations, deep sequence models), and utilities for fetching
(or generating) recommendation datasets, it aims to be a tool for rapid exploration
and prototyping of new recommender models.

See the full `documentation <https://maciejkula.github.io/spotlight/>`_ for details.

Installation
~~~~~~~~~~~~

.. code-block:: python

   conda install -c maciejkula -c pytorch spotlight


Usage
~~~~~

Factorization models
====================

To fit an explicit feedback model on the MovieLens dataset:

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
   :hide:

To fit an implicit ranking model with a BPR pairwise loss on the MovieLens dataset:

.. testcode::

    from spotlight.cross_validation import random_train_test_split
    from spotlight.datasets.movielens import get_movielens_dataset
    from spotlight.evaluation import mrr_score
    from spotlight.factorization.implicit import ImplicitFactorizationModel

    dataset = get_movielens_dataset(variant='100K')

    train, test = random_train_test_split(dataset)

    model = ImplicitFactorizationModel(n_iter=3,
                                       loss='bpr')
    model.fit(train)

    mrr = mrr_score(model, test)

.. testoutput::
   :hide:


Sequential models
=================

Recommendations can be seen as a sequence prediction task: given the items a user
has interacted with in the past, what will be the next item they will interact
with? Spotlight provides a range of models and utilities for fitting next item
recommendation models, including

- pooling models, as in `YouTube recommendations <https://pdfs.semanticscholar.org/bcdb/4da4a05f0e7bc17d1600f3a91a338cd7ffd3.pdf>`_,
- LSTM models, as in `Session-based recommendations... <https://arxiv.org/pdf/1511.06939>`_, and
- causal convolution models, as in `WaveNet <https://arxiv.org/pdf/1609.03499>`_.

.. testcode::

    from spotlight.cross_validation import user_based_train_test_split
    from spotlight.datasets.synthetic import generate_sequential
    from spotlight.evaluation import sequence_mrr_score
    from spotlight.sequence.implicit import ImplicitSequenceModel

    dataset = generate_sequential(num_users=100,
                                  num_items=1000,
                                  num_interactions=10000,
                                  concentration_parameter=0.01,
                                  order=3)

    train, test = user_based_train_test_split(dataset)

    train = train.to_sequence()
    test = test.to_sequence()

    model = ImplicitSequenceModel(n_iter=3,
                                  representation='cnn',
                                  loss='bpr')
    model.fit(train)

    mrr = sequence_mrr_score(model, test)

.. testoutput::
   :hide:
  

Datasets
========

Spotlight offers a slew of popular datasets, including Movielens 100K, 1M, 10M, and 20M.
It also incorporates utilities for creating synthetic datasets. For example, `generate_sequential`
generates a Markov-chain-derived interaction dataset, where the next item a user chooses is
a function of their previous interactions:

.. testcode::

    from spotlight.datasets.synthetic import generate_sequential

    # Concentration parameter governs how predictable the chain is;
    # order determins the order of the Markov chain.
    dataset = generate_sequential(num_users=100,
                                  num_items=1000,
                                  num_interactions=10000,
                                  concentration_parameter=0.01,
                                  order=3)

.. testoutput::
   :hide:


Examples
~~~~~~~~

1. `Rating prediction on the Movielens dataset <https://github.com/maciejkula/spotlight/tree/master/examples/movielens_explicit>`_.
2. `Using causal convolutions for sequence recommendations <https://github.com/maciejkula/spotlight/tree/master/examples/movielens_sequence>`_.
3. `Bloom embedding layers <https://github.com/maciejkula/spotlight/tree/master/examples/bloom_embeddings>`_.


How to cite
~~~~~~~~~~~

Please cite Spotlight if it helps your research. You can use the following BibTeX entry:

.. code-block::

   @misc{kula2017spotlight,
     title={Spotlight},
     author={Kula, Maciej},
     year={2017},
     publisher={GitHub},
     howpublished={\url{https://github.com/maciejkula/spotlight}},
   }


Contributing
~~~~~~~~~~~~

Spotlight is meant to be extensible: pull requests are welcome. Development progress is tracked on `Trello <https://trello.com/b/G5iFgS1W/spotlight>`_: have a look at the outstanding tickets to get an idea of what would be a useful contribution.

We accept implementations of new recommendation models into the Spotlight model zoo: if you've just published a paper describing your new model, or have an implementation of a model from the literature, make a PR!
