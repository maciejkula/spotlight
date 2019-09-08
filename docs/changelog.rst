=========
Changelog
=========

v0.1.6 (2018-05-20)
-----------------------

Other
~~~~~

* Bump to PyTorch v1.1.0.

v0.1.5 (2018-05-20)
-------------------

Other
~~~~~

* Migration to PyTorch v0.4.0.

v0.1.4 (2018-02-18)
-------------------

Fixed
~~~~~

* Bugs due to use of int32s instead of int64s on Windows (thanks to Roman Yurchak).

Other
~~~~~

* Added Appveyor for Windows CI (thanks to Roman Yurchak).

v0.1.3 (2017-12-14)
-------------------

Added
~~~~~

* Goodbooks dataset.
* Mixture-of-tastes representations.

Changed
~~~~~~~

* Raise ValueError if loss becomes NaN or 0.
* Updated to work with PyTorch 0.3.0.

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
