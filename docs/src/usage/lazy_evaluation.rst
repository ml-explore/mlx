.. _lazy_eval:

Lazy Evaluation
===============

.. currentmodule:: mlx.core

Why Lazy Evaluation
-------------------

When you perform operations in MLX, no computation actually happens. Instead a
compute graph is recorded. The actual computation only happens if an
:func:`eval` is performed.

MLX uses lazy evaluation because it has some nice features:

Transforming Compute Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lazy evaluation let's us record a compute graph without actually doing any
computations. This is useful for function transformations like :func:`grad` and
:func:`vmap` and graph optimizations like :func:`simplify`.

Currently, MLX does not compile and rerun compute graphs. They are all
generated dynamically. However, lazy evaluation makes it much easier to
integrate compilation for future performance enhancements.

Only Compute What You Use
^^^^^^^^^^^^^^^^^^^^^^^^^

In MLX you do not need to worry as much about computing outputs that are never
used. For example:

.. code-block:: python

  def fun(x):
    a = fun1(x)
    b = expensive_fun(a)
    return a, b

  y, _ = fun(x)

Here, we never actually compute the output of ``expensive_fun``. One should
still use this pattern with care, as the graph of ``expensive_fun`` is built,
and that has some cost associated to it.

Similarly lazy evaluation can be beneficial for saving memory while keeping
code simple. Say you have a very large model ``Model`` derived from
:obj:`mlx.nn.Module`. You can instantiate this model with ``model = Model()``.
Typically, this will initialize all of the weights as ``float32``, but the
initialization does not actually compute anything. If you then update the model
with ``float16`` weights, your maximum consumed memory will be about half that
required without lazy computation. 

This pattern is simple to do in MLX thanks to lazy computation:

.. code-block:: python

  model = Model() # no memory used yet
  model.load_weights("weights_fp16.safetensors")

When to Evaluate
----------------

A common question is when to use :func:`eval`. The trade-off is between
letting graphs get too large and not batching enough useful work.

For example:

.. code-block:: python

  for _ in range(100):
       a = a + b
       mx.eval(a)
       b = b * 2
       mx.eval(b)

This is a bad idea because there is some fixed overhead with each graph
evaluation. On the other hand, there is some slight overhead which grows with
the compute graph size, so extremely large graphs (while computationally
correct), can be also be costly.

Luckily a wide range of compute graph sizes work pretty well with MLX:
anything from a few tens of operations to many thousands of operations per
eval should be okay.

Most numerical computations have an iterative outer loop (e.g. the iteration in
stochastic gradient descent). A natural and usually efficient place to use
:func:`eval` is at each iteration of this outer loop.

Here is a more concrete example:

.. code-block:: python

   for batch in dataset:

       # Nothing has been evaluated yet
       loss, grad = nn.value_and_grad(model, batch)

       # Still nothing has been evaluated
       optimizer.update(model, grad)

       # Evaluate the loss and the new parameters which will
       # run the full gradient computation and optimizer update
       mx.eval(loss, model.parameters())


An important behavior to be aware of is when the graph will be implicitly
evaluated. Anytime you ``print`` and array, convert it to an
:obj:`numpy.ndarray`, or otherwise access it's memory via :obj:`memoryview`,
the graph will be evaluated. Saving arrays via :func:`save` (or any other MLX
saving functions) will also evaluate the array.


Calling :func:`array.item` on a scalar array will also evaluate it. In the
example above, printing the loss ``print(loss)`` or adding the loss scalar to
a list ``losses.append(loss.item())`` will cause a graph evaluation. If you
put these lines before evaluating the model's updated parameters this will be
a partial evaluation computing only the forward pass.

.. warning::

  Using arrays for control-flow will cause an evaluation.

Here is an example:

.. code-block:: python

   def fun(x):
       h, y = first_layer(x)
       if y > 0:  # An evaluation is done here!
           z  = second_layer_a(h)
       else:
           z  = second_layer_b(h)
       return z

Using arrays for control flow should be done with care. The above example works
and can even be used with gradient transformations. However, this can be very
inefficient if evaluations are done too frequently.
