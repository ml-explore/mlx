.. _indexing:

Indexing Arrays
===============

.. currentmodule:: mlx.core

For the most part, indexing an MLX :obj:`array` works the same as indexing a
NumPy :obj:`numpy.ndarray`. See the `NumPy documentation
<https://numpy.org/doc/stable/user/basics.indexing.html>`_ for more details on
how that works.

For example, you can use regular integers and slices (:obj:`slice`) to index arrays:

.. code-block:: shell

  >>> arr = mx.arange(10)
  >>> arr[3]
  array(3, dtype=int32)
  >>> arr[-2]  # negative indexing works
  array(8, dtype=int32)
  >>> arr[2:8:2] # start, stop, stride
  array([2, 4, 6], dtype=int32)

For multi-dimensional arrays, the ``...`` or :obj:`Ellipsis` syntax works as in NumPy:

.. code-block:: shell

  >>> arr = mx.arange(8).reshape(2, 2, 2)
  >>> arr[:, :, 0]
  array(3, dtype=int32)
  array([[0, 2],
         [4, 6]], dtype=int32
  >>> arr[..., 0]
  array([[0, 2],
         [4, 6]], dtype=int32

You can index with ``None`` to create a new axis:

.. code-block:: shell

  >>> arr = mx.arange(8)
  >>> arr.shape
  [8]
  >>> arr[None].shape
  [1, 8]


You can also use an :obj:`array` to index another :obj:`array`:

.. code-block:: shell

  >>> arr = mx.arange(10)
  >>> idx = mx.array([5, 7]) 
  >>> arr[idx]
  array([5, 7], dtype=int32)

Mixing and matching integers, :obj:`slice`, ``...``, and :obj:`array` indices
works just as in NumPy.

Other functions which may be useful for indexing arrays are :func:`take` and
:func:`take_along_axis`.

Differences from NumPy
----------------------

.. Note::

  MLX indexing is different from NumPy indexing in two important ways:

  * Indexing does not perform bounds checking. Indexing out of bounds is
    undefined behavior.
  * Boolean mask based indexing is not yet supported.

The reason for the lack of bounds checking is that exceptions cannot propagate
from the GPU. Performing bounds checking for array indices before launching the
kernel would be extremely inefficient.

Indexing with boolean masks is something that MLX may support in the future. In
general, MLX has limited support for operations for which outputs
*shapes* are dependent on input *data*. Other examples of these types of
operations which MLX does not yet support include :func:`numpy.nonzero` and the
single input version of :func:`numpy.where`.

In Place Updates 
----------------

In place updates to indexed arrays are possible in MLX. For example:

.. code-block:: shell

  >>> a = mx.array([1, 2, 3])
  >>> a[2] = 0
  >>> a
  array([1, 2, 0], dtype=int32)

Just as in NumPy, in place updates will be reflected in all references to the
same array:

.. code-block:: shell

  >>> a = mx.array([1, 2, 3])
  >>> b = a
  >>> b[2] = 0
  >>> b
  array([1, 2, 0], dtype=int32)
  >>> a
  array([1, 2, 0], dtype=int32)

Transformations of functions which use in-place updates are allowed and work as
expected. For example:

.. code-block:: python

   def fun(x, idx):
       x[idx] = 2.0
       return x.sum()

   dfdx = mx.grad(fun)(mx.array([1.0, 2.0, 3.0]), mx.array([1]))
   print(dfdx)  # Prints: array([1, 0, 1], dtype=float32)

In the above ``dfdx`` will have the correct gradient, namely zeros at ``idx``
and ones elsewhere.
