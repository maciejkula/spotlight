===================
Model serialization
===================

Saving and loading the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To save a ``spotlight`` model, you can simply use ``torch`` serialization utilities::

  torch.save(spotlight_model, PATH)

and then::

  spotlight_model = torch.load(PATH)
