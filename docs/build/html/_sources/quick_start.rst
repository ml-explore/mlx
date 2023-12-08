Quick Start Guide
=================


Basics
------

.. currentmodule:: mlx.core

Import ``mlx.core`` and make an :class:`array`:

.. code-block:: python

  >> import mlx.core as mx
  >> a = mx.array([1, 2, 3, 4])
  >> a.shape
  [4]
  >> a.dtype
  int32
  >> b = mx.array([1.0, 2.0, 3.0, 4.0])
  >> b.dtype
  float32

Operations in MLX are lazy. The outputs of MLX operations are not computed
until they are needed. To force an array to be evaluated use
:func:`eval`.  Arrays will automatically be evaluated in a few cases. For
example, inspecting a scalar with :meth:`array.item`, printing an array,
or converting an array from :class:`array` to :class:`numpy.ndarray` all
automatically evaluate the array.

.. code-block:: python

  >> c = a + b    # c not yet evaluated
  >> mx.eval(c)  # evaluates c
  >> c = a + b
  >> print(c)     # Also evaluates c
  array([2, 4, 6, 8], dtype=float32)
  >> c = a + b
  >> import numpy as np
  >> np.array(c)   # Also evaluates c
  array([2., 4., 6., 8.], dtype=float32)

Function and Graph Transformations
----------------------------------

MLX has standard function transformations like :func:`grad` and :func:`vmap`.
Transformations can be composed arbitrarily. For example
``grad(vmap(grad(fn)))`` (or any other composition) is allowed.

.. code-block:: python

  >> x = mx.array(0.0)
  >> mx.sin(x)
  array(0, dtype=float32)
  >> mx.grad(mx.sin)(x)
  array(1, dtype=float32)
  >> mx.grad(mx.grad(mx.sin))(x)
  array(-0, dtype=float32)

Other gradient transformations include :func:`vjp` for vector-Jacobian products
and :func:`jvp` for Jacobian-vector products.

Use :func:`value_and_grad` to efficiently compute both a function's output and
gradient with respect to the function's input. 
