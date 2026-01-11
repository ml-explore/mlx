.. _distributed:

Distributed
-----------

Helper Routines
^^^^^^^^^^^^^^^

The :code:`mlx.nn.layers.distributed` package contains helpful routines for creating sharded layers easily from existing :class:`Modules <mlx.nn.Module>`.

.. currentmodule:: mlx.nn.layers.distributed
.. autosummary::
   :toctree: _autosummary

   shard_linear
   shard_inplace

Layers
^^^^^^

The :code:`mlx.nn` package contains helpful layers for creating distributed models with sharded parameters.

.. currentmodule:: mlx.nn
.. autosummary::
   :toctree: _autosummary
   :template: nn-module-template.rst

   AllToShardedLinear
   ShardedToAllLinear
   QuantizedAllToShardedLinear
   QuantizedShardedToAllLinear
