=========
Changelog
=========

unreleased (unreleased)
-----------------------

Added
~~~~~

* Goodbooks dataset.

Changed
~~~~~~~

* Raise ValueError if loss becomes NaN or 0.

v0.1.2 (2017-09-10)
-------------------

Added
~~~~~

* :class:`spotlight.layers.BloomEmbedding`: bloom embedding layers that reduce the number of
  parameters required by hashing embedding indices into some fixed smaller dimensionality,
  following Serrà, Joan, and Alexandros Karatzoglou. "Getting deep recommenders fit: Bloom
  embeddings for sparse binary input/output networks."
* ``sequence_mrr_score`` now accepts an option that excludes previously seen items from scoring.

Changed
~~~~~~~

* ``optimizer`` arguments is now ``optimizer_func``. It accepts a function that takes a single argument (list of model parameters) and return a PyTorch optimizer (thanks to Ethan Rosenthal).
* ``fit`` calls will resume from previous model state when called repeatedly (Ethan Rosenthal).
* Updated to work with PyTorch v0.2.0.

Fixed
~~~~~

* Factorization predict APIs now work as advertised in the documentation.
