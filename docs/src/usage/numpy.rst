.. _numpy:

Conversion to NumPy and Other Frameworks
========================================

MLX array supports conversion between other frameworks with either:

* The `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_.
* `DLPack <https://dmlc.github.io/dlpack/latest/>`_.

Let's convert an array to NumPy and back.

.. code-block:: python

  import mlx.core as mx
  import numpy as np

  a = mx.arange(3)
  b = np.array(a) # copy of a
  c = mx.array(b) # copy of b

.. note::

    Since NumPy does not support ``bfloat16`` arrays, you will need to convert
    to ``float16`` or ``float32`` first: ``np.array(a.astype(mx.float32))``.
    Otherwise, you will receive an error like: ``Item size 2 for PEP 3118
    buffer format string does not match the dtype V item size 0.``

By default, NumPy copies data to a new array. This can be prevented by creating
an array view:

.. code-block:: python

  a = mx.arange(3)
  a_view = np.array(a, copy=False)
  print(a_view.flags.owndata) # False
  a_view[0] = 1
  print(a[0].item()) # 1

.. note::

    NumPy arrays with type ``float64`` will be default converted to MLX arrays
    with type ``float32``.

A NumPy array view is a normal NumPy array, except that it does not own its
memory. This means writing to the view is reflected in the original array.

While this is quite powerful to prevent copying arrays, it should be noted that
external changes to the memory of arrays cannot be reflected in gradients.

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
However, this modification is not reflected in the gradient, as seen in the
last line outputting ``1.0``, representing the gradient of the sum operation
alone.  The squaring of ``x`` occurs externally to MLX, meaning that no
gradient is incorporated.  It's important to note that a similar issue arises
during array conversion and copying.  For instance, a function defined as
``mx.array(np.array(x)**2).sum()`` would also result in an incorrect gradient,
even though no in-place operations on MLX memory are executed.

PyTorch
-------

PyTorch supports DLPack inputs and can import MLX arrays directly.
MLX can also import PyTorch tensors through DLPack with ``mx.array`` or
``mx.from_dlpack``.

.. code-block:: python

  import mlx.core as mx
  import torch

  a = mx.arange(3)
  b = torch.tensor(a)
  c = mx.array(b)

Creating an MLX array from a CPU tensor copies the data into MLX-owned storage.
The arrays do not share memory:

.. code-block:: python

  b = torch.arange(3)
  c = mx.array(b)

  b += 10
  print(c.tolist()) # [0, 1, 2]

Metal DLPack inputs are different. If a PyTorch MPS tensor is passed to
``mx.array`` or to ``mx.from_dlpack`` with ``copy=None`` or ``copy=False``, MLX
imports the underlying Metal buffer without copying it. The PyTorch tensor and
the MLX array then share the same storage. MLX arrays exported to PyTorch with
DLPack are also shared without a copy.

Since the buffer is shared across frameworks, synchronization has to be managed
explicitly. After PyTorch writes to an MPS tensor, call
``torch.mps.synchronize()`` before reading the shared data from MLX. After MLX
writes to the shared array, call ``mx.eval`` on the MLX result before reading
the shared data from PyTorch. Without these synchronization points, the other
framework may read the shared buffer before the producer has finished writing,
so it can observe stale data.

.. code-block:: python

  b = torch.arange(3, device="mps", dtype=torch.float32)
  torch.mps.synchronize()
  c = mx.array(b) # zero-copy Metal DLPack import

  b.add_(10)
  torch.mps.synchronize()
  print(c.tolist()) # [10.0, 11.0, 12.0]

Updates made by MLX can also be observed from PyTorch after the MLX computation
has been evaluated:

.. code-block:: python

  b = torch.arange(3, device="mps", dtype=torch.float32)
  torch.mps.synchronize()
  c = mx.array(b)

  c += 10
  mx.eval(c)
  print(b.cpu()) # tensor([10., 11., 12.])

For MLX arrays exported to PyTorch, the share is tied to the exported buffer.
MLX updates after export may rebind the MLX array to a new buffer, while the
PyTorch tensor continues to reference the exported buffer.

Use ``mx.from_dlpack`` when you need to control the copy behavior. Specifying
``copy=True`` asks MLX to create a new array instead of sharing the Metal
buffer:

.. code-block:: python

  b = torch.arange(3, device="mps", dtype=torch.float32)
  torch.mps.synchronize()
  c = mx.from_dlpack(b, copy=True)

  b.add_(10)
  torch.mps.synchronize()
  print(c.tolist()) # [0.0, 1.0, 2.0]

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

TensorFlow supports the buffer protocol, but it requires an explicit
:obj:`memoryview`.

.. code-block:: python

  import mlx.core as mx
  import tensorflow as tf

  a = mx.arange(3)
  b = tf.constant(memoryview(a))
  c = mx.array(b)
