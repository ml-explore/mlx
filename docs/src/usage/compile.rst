.. _compile:

Compilation
===========

.. currentmodule:: mlx.core

MLX has a :func:`compile` function transformation which compiles computation
graphs. Function compilation results in smaller graphs by merging common work
and fusing certain operations. In many cases this can lead to big improvements
in run-time and memory use.

Getting started with :func:`compile` is simple, but there are some edge cases
that are good to be aware of for more complex graphs and advanced usage.

Basics of Compile
-----------------

Let's start with a simple example:

.. code-block:: python

  def fun(x, y):
      return mx.exp(-x) + y

  x = mx.array(1.0)
  y = mx.array(2.0)

  # Regular call, no compilation
  # Prints: array(2.36788, dtype=float32)
  print(fun(x, y))

  # Compile the function
  compiled_fun = mx.compile(fun)

  # Prints: array(2.36788, dtype=float32) 
  print(compiled_fun(x, y))

The output of both the regular function and the compiled function is the same
up to numerical precision.
   
The first time you call a compiled function, MLX will build the compute
graph, optimize it, and generate and compile code. This can be relatively
slow. However, MLX will cache compiled functions, so calling a compiled
function multiple times will not initiate a new compilation. This means you
should typically compile functions that you plan to use more than once.

.. code-block:: python

  def fun(x, y):
      return mx.exp(-x) + y

  x = mx.array(1.0)
  y = mx.array(2.0)

  compiled_fun = mx.compile(fun)

  # Compiled here
  compiled_fun(x, y)

  # Not compiled again
  compiled_fun(x, y)

  # Not compiled again
  mx.compile(fun)(x, y)

There are some important cases to be aware of that can cause a function to
be recompiled:

* Changing the shape or number of dimensions
* Changing the type of any of the inputs
* Changing the number of inputs to the function

In certain cases only some of the compilation stack will be rerun (for
example when changing the shapes) and in other cases the full compilation
stack will be rerun (for example when changing the types). In general you
should avoid compiling functions too frequently.

Another idiom to watch out for is compiling functions which get created and
destroyed frequently. This can happen, for example, when compiling an anonymous
function in a loop:

.. code-block:: python

  a = mx.array(1.0)
  # Don't do this, compiles lambda at each iteration
  for _ in range(5):
      mx.compile(lambda x: mx.exp(mx.abs(x)))(a)

Example Speedup
---------------

The :func:`mlx.nn.gelu` is a nonlinear activation function commonly used with
Transformer-based models. The implementation involves several unary and binary
element-wise operations:

.. code-block:: python

  def gelu(x):  
      return x * (1 + mx.erf(x / math.sqrt(2))) / 2

If you use this function with small arrays, it will be overhead bound. If you
use it with large arrays it will be memory bandwidth bound.  However, all of
the operations in the ``gelu`` are fusible into a single kernel with
:func:`compile`. This can speedup both cases considerably.

Let's compare the runtime of the regular function versus the compiled
function. We'll use the following timing helper which does a warm up and
handles synchronization:

.. code-block:: python

  import time

  def timeit(fun, x):
      # warm up
      for _ in range(10):
          mx.eval(fun(x))

      tic = time.perf_counter()
      for _ in range(100):
          mx.eval(fun(x))
      toc = time.perf_counter()
      tpi = 1e3 * (toc - tic) / 100
      print(f"Time per iteration {tpi:.3f} (ms)")


Now make an array, and benchmark both functions:

.. code-block:: python

  x = mx.random.uniform(shape=(32, 1000, 4096))
  timeit(nn.gelu, x)
  timeit(mx.compile(nn.gelu), x)

On an M1 Max the times are 15.5 and 3.1 milliseconds. The compiled ``gelu`` is
five times faster.

.. note::

  As of the latest MLX, CPU functions are not fully compiled. Compiling CPU
  functions can still be helpful, but won't typically result in as large a
  speedup as compiling operations that run on the GPU.


Debugging
---------

When a compiled function is first called, it is traced with placeholder
inputs. This means you can't evaluate arrays (for example to print their
contents) inside compiled functions.

.. code-block:: python

  @mx.compile
  def fun(x):
      z = -x
      print(z)  # Crash
      return mx.exp(z)

  fun(mx.array(5.0))

For debugging, inspecting arrays can be helpful. One way to do that is to
globally disable compilation using the :func:`disable_compile` function or
``MLX_DISABLE_COMPILE`` flag. For example the following is okay even though
``fun`` is compiled:

.. code-block:: python

  @mx.compile
  def fun(x):
      z = -x
      print(z) # Okay
      return mx.exp(z)

  mx.disable_compile()
  fun(mx.array(5.0))


Pure Functions
--------------

Compiled functions are intended to be *pure*; that is they should not have side
effects. For example:

.. code-block:: python

  state = []

  @mx.compile
  def fun(x, y):
      z = x + y
      state.append(z)
      return mx.exp(z)

  fun(mx.array(1.0), mx.array(2.0))
  # Crash!
  print(state)

After the first call of ``fun``, the ``state`` list will hold a placeholder
array. The placeholder does not have any data; it is only used to build the
computation graph. Printing such an array results in a crash.

You have two options to deal with this. The first option is to simply return
``state`` as an output:

.. code-block:: python

   state = []

   @mx.compile
   def fun(x, y):
      z = x + y
      state.append(z)
      return mx.exp(z), state

    _, state = fun(mx.array(1.0), mx.array(2.0))
    # Prints [array(3, dtype=float32)]
    print(state)

In some cases returning updated state can be pretty inconvenient. Hence,
:func:`compile` has a parameter to capture implicit outputs:

.. code-block:: python

  from functools import partial

  state = []

  # Tell compile to capture state as an output
  @partial(mx.compile, outputs=state)
  def fun(x, y):
      z = x + y
      state.append(z)
      return mx.exp(z), state

  fun(mx.array(1.0), mx.array(2.0))
  # Prints [array(3, dtype=float32)]
  print(state)

This is particularly useful for compiling a function which includes an update
to a container of arrays, as is commonly done when training the parameters of a
:class:`mlx.nn.Module`.

Compiled functions will also treat any inputs not in the parameter list as
constants. For example:

.. code-block:: python

  state = [mx.array(1.0)]

  @mx.compile
  def fun(x):
      return x + state[0]

  # Prints array(2, dtype=float32)
  print(fun(mx.array(1.0)))

  # Update state
  state[0] = mx.array(5.0)

  # Still prints array(2, dtype=float32)
  print(fun(mx.array(1.0)))

In order to have the change of state reflected in the outputs of ``fun`` you
again have two options. The first option is to simply pass ``state`` as input
to the function. In some cases this can be pretty inconvenient. Hence,
:func:`compile` also has a parameter to capture implicit inputs:

.. code-block:: python

  from functools import partial
  state = [mx.array(1.0)]

  # Tell compile to capture state as an input
  @partial(mx.compile, inputs=state)
  def fun(x):
      return x + state[0]

  # Prints array(2, dtype=float32)
  print(fun(mx.array(1.0)))

  # Update state
  state[0] = mx.array(5.0)

  # Prints array(6, dtype=float32)
  print(fun(mx.array(1.0)))


Compiling Training Graphs 
-------------------------

This section will step through how to use :func:`compile` with a simple example
of a common setup: training a model with :obj:`mlx.nn.Module` using an
:obj:`mlx.optimizers.Optimizer` with state. We will show how to compile the
full forward, backward, and update with :func:`compile`.

To start, here is the simple example without any compilation:

.. code-block:: python 

  import mlx.core as mx
  import mlx.nn as nn
  import mlx.optimizers as optim

  # 4 examples with 10 features each
  x = mx.random.uniform(shape=(4, 10))

  # 0, 1 targets
  y = mx.array([0, 1, 0, 1])

  # Simple linear model
  model = nn.Linear(10, 1)

  # SGD with momentum
  optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)

  def loss_fn(model, x, y):
      logits = model(x).squeeze()
      return nn.losses.binary_cross_entropy(logits, y)

  loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

  # Perform 10 steps of gradient descent
  for it in range(10):
      loss, grads = loss_and_grad_fn(model, x, y)
      optimizer.update(model, grads)
      mx.eval(model.parameters(), optimizer.state)

To compile the update we can put it all in a function and compile it with the
appropriate input and output captures. Here's the same example but compiled:

.. code-block:: python 

  import mlx.core as mx
  import mlx.nn as nn
  import mlx.optimizers as optim
  from functools import partial

  # 4 examples with 10 features each
  x = mx.random.uniform(shape=(4, 10))

  # 0, 1 targets
  y = mx.array([0, 1, 0, 1])

  # Simple linear model
  model = nn.Linear(10, 1)

  # SGD with momentum
  optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)

  def loss_fn(model, x, y):
      logits = model(x).squeeze()
      return nn.losses.binary_cross_entropy(logits, y)

  # The state that will be captured as input and output
  state = [model.state, optimizer.state]
      
  @partial(mx.compile, inputs=state, outputs=state)
  def step(x, y):
      loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
      loss, grads = loss_and_grad_fn(model, x, y)
      optimizer.update(model, grads)
      return loss

  # Perform 10 steps of gradient descent
  for it in range(10):
      loss = step(x, y)
      # Evaluate the model and optimizer state
      mx.eval(state)
      print(loss)


.. note::

  If you are using a module which performs random sampling such as
  :func:`mlx.nn.Dropout`, make sure you also include ``mx.random.state`` in the
  ``state`` captured by :func:`compile`, i.e. ``state = [model.state,
  optimizer.state, mx.random.state]``.


.. note::

   For more examples of compiling full training graphs checkout the  `MLX
   Examples <https://github.com/ml-explore/mlx-examples>`_ GitHub repo.

Transformations with Compile
----------------------------

In MLX function transformations are composable. You can apply any function
transformation to the output of any other function transformation. For more on
this, see the documentation on :ref:`function transforms
<function_transforms>`.

Compiling transformed functions works just as expected:

.. code-block:: python

  grad_fn = mx.grad(mx.exp)

  compiled_grad_fn = mx.compile(grad_fn)

  # Prints: array(2.71828, dtype=float32)
  print(grad_fn(mx.array(1.0)))

  # Also prints: array(2.71828, dtype=float32)
  print(compiled_grad_fn(mx.array(1.0)))

.. note::

   In order to compile as much as possible, a transformation of a compiled
   function will not by default be compiled. To compile the transformed
   function simply pass it through :func:`compile`. 

You can also compile functions which themselves call compiled functions. A
good practice is to compile the outer most function to give :func:`compile`
the most opportunity to optimize the computation graph:

.. code-block:: python

  @mx.compile
  def inner(x):
      return mx.exp(-mx.abs(x))

  def outer(x):
      inner(inner(x))

  # Compiling the outer function is good to do as it will likely
  # be faster even though the inner functions are compiled
  fun = mx.compile(outer)
