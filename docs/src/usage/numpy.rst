.. _numpy:

Conversion to NumPy and Other Frameworks
========================================

MLX array implements the `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_.
Let's convert an array to NumPy and back.

.. code-block:: python

  import mlx.core as mx
  import numpy as np

  a = mx.arange(3)
  b = np.array(a) # copy of a
  c = mx.array(b) # copy of b

.. note::

    Since NumPy does not support ``bfloat16`` arrays, you will need to convert to ``float16`` or ``float32`` first:
    ``np.array(a.astype(mx.float32))``.
    Otherwise, you will receive an error like: ``Item size 2 for PEP 3118 buffer format string does not match the dtype V item size 0.``

By default, NumPy copies data to a new array. This can be prevented by creating an array view:

.. code-block:: python

  a = mx.arange(3)
  a_view = np.array(a, copy=False)
  print(a_view.flags.owndata) # False
  a_view[0] = 1
  print(a[0].item()) # 1

A NumPy array view is a normal NumPy array, except that it does not own its memory.
This means writing to the view is reflected in the original array.

While this is quite powerful to prevent copying arrays, it should be noted that external changes to the memory of arrays cannot be reflected in gradients.

Let's demonstrate this in an example:

.. code-block:: python

  def f(x):
      x_view = np.array(x, copy=False)
      x_view[:] *= x_view # modify memory without telling mx
      return x.sum()

  x = mx.array([3.0])
  y, df = mx.value_and_grad(f)(x)
  print("f(x) = x² =", y.item()) # 9.0
  print("f'(x) = 2x !=", df.item()) # 1.0


The function ``f`` indirectly modifies the array ``x`` through a memory view.
However, this modification is not reflected in the gradient, as seen in the last line outputting ``1.0``,
representing the gradient of the sum operation alone.
The squaring of ``x`` occurs externally to MLX, meaning that no gradient is incorporated.
It's important to note that a similar issue arises during array conversion and copying.
For instance, a function defined as ``mx.array(np.array(x)**2).sum()`` would also result in an incorrect gradient,
even though no in-place operations on MLX memory are executed.

PyTorch
-------

.. warning:: 

   PyTorch Support for :obj:`memoryview` is experimental and can break for
   multi-dimensional arrays. Casting to NumPy first is advised for now.

PyTorch supports the buffer protocol, but it requires an explicit :obj:`memoryview`.

.. code-block:: python

  import mlx.core as mx
  import torch

  a = mx.arange(3)
  b = torch.tensor(memoryview(a))
  c = mx.array(b.numpy())

Conversion from PyTorch tensors back to arrays must be done via intermediate NumPy arrays with ``numpy()``.

JAX
---
JAX fully supports the buffer protocol.

.. code-block:: python

  import mlx.core as mx
  import jax.numpy as jnp

  a = mx.arange(3)
  b = jnp.array(a)
  c = mx.array(b)

TensorFlow
----------

TensorFlow supports the buffer protocol, but it requires an explicit :obj:`memoryview`.

.. code-block:: python

  import mlx.core as mx
  import tensorflow as tf

  a = mx.arange(3)
  b = tf.constant(memoryview(a))
  c = mx.array(b)
