=========
Changelog
=========

v0.1.2 (unreleased)
-------------------

Changed
~~~~~~~

* ``optimizer`` arguments is now ``optimizer_func``. It accepts a function that takes a single argument (list of model parameters) and return a PyTorch optimizer (thanks to Ethan Rosenthal).
* ``fit`` calls will resume from previous model state when called repeatedly (Ethan Rosenthal).

Fixed
~~~~~

* Factorization predict APIs now work as advertised in the documentation.

