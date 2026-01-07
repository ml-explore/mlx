Tensor Parallelism
==================

MLX enables efficient implementation of tensor parallelism *(TP)* through its implementation of distributed layers. In this example we will explore what these layers are and create a small inference script for Llama family transformer models using MLX tensor parallelism.

MLX Sharded Layers
------------------

:class:`AllToShardedLinear <mlx.nn.AllToShardedLinear>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Column-wise tensor parallelism. This layer replicates a common input and shards the weight matrix along the output dimension (column-wise) across all devices in the :class:`mlx.core.distributed.Group`. The layer produces a sharded output.

For example, consider an :class:`mlx.nn.AllToShardedLinear` layer with ``input_dims=2`` and ``output_dims=2``, a batched input of shape ``(4, 2)``, and a device group with 2 devices. The layer shards the weight matrix column-wise across the two devices, where each device receives the full input and computes a partial output.

.. raw:: html

    <div>
      <img src="../_static/distributed/AllToShardedLinear.png" alt="column-wise tensor parallelism" style="width: 100%">
      <p style="font-size: 0.85em; margin-top: 0.5em;"><small>Check out <a href="https://huggingface.co/spaces/gxa33/ultrascale-playbook?section=tensor_parallelism_in_a_transformer_block">huggingface ultrascale-playbook</a> to learn about TP more in depth.</small></p>
    </div>

This layer does not automatically gather all outputs from each device. This is an intended and :ref:`useful design choice <useful_design_choices>`.

:class:`ShardedToAllLinear <mlx.nn.ShardedToAllLinear>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Row-wise tensor parallelism. This layer expects inputs that are sharded along the feature dimension (column-wise) and shards the weight matrix along the input dimension (row-wise) across all devices in the :class:`mlx.core.distributed.Group`. The layer automatically aggregates the results using :class:`mlx.core.distributed.all_sum`, so all devices in the group will have the same result.

For example, consider an :class:`mlx.nn.ShardedToAllLinear` layer with ``input_dims=2`` and ``output_dims=2``, a batched input of shape ``(4, 2)``, and a device group with 2 devices. The layer shards the weight matrix row-wise and the input column-wise across the two devices. Each device computes a ``(4,2)`` output, which is then aggregated with all other device outputs to get layer output.

.. raw:: html

    <div>
      <img src="../_static/distributed/ShardedToAllLinear.png" alt="row-wise tensor parallelism" style="width: 100%">
    </div>

This layer does not automatically shard the inputs along the batch dimension for you. It is necessary to create a "partial" input structure to feed into the layer. This is an intended and :ref:`useful design choice <useful_design_choices>`.

We can create partial inputs based on rank. For example, for ``batch_size=1024``, we can create sharded inputs in the following manner:

.. code-block:: python

  world = mx.distributed.init()
  part = (
      slice(None), # batch dimension keep everything in column
      slice(
          world.rank() * 1024 // world.size(), # start
          (world.rank() + 1) * 1024 // world.size(), # end
      ),
  )

  layer = nn.ShardedToAllLinear(1024, 1024, bias=False) # initialize the layer
  y = layer(x[part]) # process sharded input

This code splits the 1024 input features into ``world.size()`` different groups which are assigned continously based on ``world.rank()``. More information about distributed communication can be found in the :doc:`Distributed Communication <usage/distributed>` page. 

:class:`QuantizedAllToShardedLinear <mlx.nn.QuantizedAllToShardedLinear>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`QuantizedShardedToAllLinear <mlx.nn.QuantizedShardedToAllLinear>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _useful_design_choices:
Useful Design Choices
^^^^^^^^^^^^^^^^^^^^^

There are design choices made above related to when things are done automatically that are done on purpose to make model training / inference easier.

Column-wise and row-wise tensor parallel layers naturally go together due to the output of the column-wise TP layer being the exact size needed for the sharded input of a row-wise TP layer. This removes the need for an intermediary gather step between the layers, reducing communication overhead.

This is why AllToShardedLinear does not aggregate results automatically and why ShardedToAllLinear does not shard inputs automatically. It is so that they can be placed in succesive order and work together easily.

We can demonstrate this through a simple model using our two types of distributed layers.

.. code-block:: python

  x = mx.array() # (4, 2) model input: batch size 4, feature size 2

  l1 = nn.AllToShardedLinear(2, 2, bias=False)   # initialize the layer
  l1_out = l1(x) # (4, 1) output

  l2 = nn.ShardedToAllLinear(2, 2, bias=False)
  l2_out = l2(l1_out) # (4, 2) output

.. raw:: html

    <div>
      <img src="../_static/distributed/ColumnRowTP.png" alt="column-wise tensor parallelism" style="width: 100%">
      <p style="font-size: 0.85em; margin-top: 0.5em;"><small>A visualization of the simple MLX model using column-wise then row-wise tensor parallelism across 2 devices.</small></p>
    </div>



