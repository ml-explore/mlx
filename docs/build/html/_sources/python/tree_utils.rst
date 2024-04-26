.. _utils:

Tree Utils
==========

In MLX we consider a python tree to be an arbitrarily nested collection of
dictionaries, lists and tuples without cycles. Functions in this module that
return python trees will be using the default python ``dict``, ``list`` and
``tuple`` but they can usually process objects that inherit from any of these.

.. note::
   Dictionaries should have keys that are valid python identifiers.

.. currentmodule:: mlx.utils

.. autosummary:: 
  :toctree: _autosummary

   tree_flatten
   tree_unflatten
   tree_map
   tree_map_with_path
