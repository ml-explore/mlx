.. _nn_distributed:

Distributed
-----------

Helper Routines
^^^^^^^^^^^^^^^

The :code:`mlx.nn.layers.distributed` package contains helpful routines to 
create sharded layers from existing :class:`Modules <mlx.nn.Module>`.

.. currentmodule:: mlx.nn.layers.distributed
.. autosummary::
   :toctree: _autosummary

   shard_linear
   shard_inplace

Layers
^^^^^^

.. currentmodule:: mlx.nn
.. autosummary::
   :toctree: _autosummary
   :template: nn-module-template.rst

   AllToShardedLinear
   ShardedToAllLinear
   QuantizedAllToShardedLinear
   QuantizedShardedToAllLinear
