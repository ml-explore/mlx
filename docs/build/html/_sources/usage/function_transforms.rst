.. _function_transforms:

Function Transforms
===================

.. currentmodule:: mlx.core

MLX uses composable function transformations for automatic differentiation,
vectorization, and compute graph optimizations. To see the complete list of
function transformations check-out the :ref:`API documentation <transforms>`.

The key idea behind composable function transformations is that every
transformation returns a function which can be further transformed.

Here is a simple example:

.. code-block:: shell

   >>> dfdx = mx.grad(mx.sin)
   >>> dfdx(mx.array(mx.pi))
   array(-1, dtype=float32)
   >>> mx.cos(mx.array(mx.pi))
   array(-1, dtype=float32)


The output of :func:`grad` on :func:`sin` is simply another function. In this
case it is the gradient of the sine function which is exactly the cosine
function. To get the second derivative you can do: 

.. code-block:: shell

   >>> d2fdx2 = mx.grad(mx.grad(mx.sin))
   >>> d2fdx2(mx.array(mx.pi / 2))
   array(-1, dtype=float32)
   >>> mx.sin(mx.array(mx.pi / 2))
   array(1, dtype=float32)

Using :func:`grad` on the output of :func:`grad` is always ok. You keep
getting higher order derivatives.

Any of the MLX function transformations can be composed in any order to any
depth. See the following sections for more information on :ref:`automatic
differentiaion <auto diff>` and :ref:`automatic vectorization <vmap>`.
For more information on :func:`compile` see the :ref:`compile documentation <compile>`.


Automatic Differentiation
-------------------------

.. _auto diff:

Automatic differentiation in MLX works on functions rather than on implicit
graphs. 

.. note::

   If you are coming to MLX from PyTorch, you no longer need functions like
   ``backward``, ``zero_grad``, and ``detach``, or properties like
   ``requires_grad``.

The most basic example is taking the gradient of a scalar-valued function as we
saw above. You can use the :func:`grad` and :func:`value_and_grad` function to
compute gradients of more complex functions. By default these functions compute
the gradient with respect to the first argument:

.. code-block:: python

   def loss_fn(w, x, y):
      return mx.mean(mx.square(w * x - y))

   w = mx.array(1.0)
   x = mx.array([0.5, -0.5])
   y = mx.array([1.5, -1.5])

   # Computes the gradient of loss_fn with respect to w:
   grad_fn = mx.grad(loss_fn)
   dloss_dw = grad_fn(w, x, y)
   # Prints array(-1, dtype=float32)
   print(dloss_dw)

   # To get the gradient with respect to x we can do:
   grad_fn = mx.grad(loss_fn, argnums=1)
   dloss_dx = grad_fn(w, x, y)
   # Prints array([-1, 1], dtype=float32)
   print(dloss_dx)


One way to get the loss and gradient is to call ``loss_fn`` followed by
``grad_fn``, but this can result in a lot of redundant work. Instead, you
should use :func:`value_and_grad`. Continuing the above example:


.. code-block:: python

   # Computes the gradient of loss_fn with respect to w:
   loss_and_grad_fn = mx.value_and_grad(loss_fn)
   loss, dloss_dw = loss_and_grad_fn(w, x, y)

   # Prints array(1, dtype=float32)
   print(loss)

   # Prints array(-1, dtype=float32)
   print(dloss_dw)


You can also take the gradient with respect to arbitrarily nested Python
containers of arrays (specifically any of :obj:`list`, :obj:`tuple`, or
:obj:`dict`).

Suppose we wanted a weight and a bias parameter in the above example. A nice
way to do that is the following:

.. code-block:: python

   def loss_fn(params, x, y):
      w, b = params["weight"], params["bias"]
      h = w * x + b 
      return mx.mean(mx.square(h - y))

   params = {"weight": mx.array(1.0), "bias": mx.array(0.0)}
   x = mx.array([0.5, -0.5])
   y = mx.array([1.5, -1.5])

   # Computes the gradient of loss_fn with respect to both the
   # weight and bias:
   grad_fn = mx.grad(loss_fn)
   grads = grad_fn(params, x, y)

   # Prints
   # {'weight': array(-1, dtype=float32), 'bias': array(0, dtype=float32)}
   print(grads)

Notice the tree structure of the parameters is preserved in the gradients.

In some cases you may want to stop gradients from propagating through a 
part of the function. You can use the :func:`stop_gradient` for that.


Automatic Vectorization
-----------------------

.. _vmap:

Use :func:`vmap` to automate vectorizing complex functions. Here we'll go
through a basic and contrived example for the sake of clarity, but :func:`vmap`
can be quite powerful for more complex functions which are difficult to optimize
by hand.

.. warning::

   Some operations are not yet supported with :func:`vmap`. If you encounter an error
   like: ``ValueError: Primitive's vmap not implemented.`` file an `issue
   <https://github.com/ml-explore/mlx/issues>`_ and include your function.
   We will prioritize including it.

A naive way to add the elements from two sets of vectors is with a loop:

.. code-block:: python

  xs = mx.random.uniform(shape=(4096, 100))
  ys = mx.random.uniform(shape=(100, 4096))

  def naive_add(xs, ys):
      return [xs[i] + ys[:, i] for i in range(xs.shape[1])]

Instead you can use :func:`vmap` to automatically vectorize the addition:

.. code-block:: python
   
   # Vectorize over the second dimension of x and the
   # first dimension of y
   vmap_add = mx.vmap(lambda x, y: x + y, in_axes=(1, 0))

The ``in_axes`` parameter can be used to specify which dimensions of the
corresponding input to vectorize over. Similarly, use ``out_axes`` to specify
where the vectorized axes should be in the outputs. 

Let's time these two different versions:

.. code-block:: python

  import timeit

  print(timeit.timeit(lambda: mx.eval(naive_add(xs, ys)), number=100))
  print(timeit.timeit(lambda: mx.eval(vmap_add(xs, ys)), number=100))

On an M1 Max the naive version takes in total ``0.390`` seconds whereas the
vectorized version takes only ``0.025`` seconds, more than ten times faster.

Of course, this operation is quite contrived. A better approach is to simply do
``xs + ys.T``, but for more complex functions :func:`vmap` can be quite handy.
