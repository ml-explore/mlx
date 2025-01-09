.. _export_usage:

Exporting Functions
===================

.. currentmodule:: mlx.core

MLX has an API to export and import functions to and from a file. This lets you
run computations written in one MLX front-end (e.g. Python) in another MLX
front-end (e.g. C++). 

This guide walks through the basics of the MLX export API with some examples.
To see the full list of functions check-out the :ref:`API documentation
<export>`.

Basics of Exporting 
-------------------

Let's start with a simple example:
 
.. code-block:: python

  def fun(x, y):
    return x + y

  x = mx.array(1.0)
  y = mx.array(1.0)
  mx.export_function("add.mlxfn", fun, x, y)

To export a function, provide sample input arrays that the function
can be called with. The data doesn't matter, but the shapes and types of the
arrays do. In the above example we exported ``fun`` with two ``float32``
scalar arrays. We can then import the function and run it:

.. code-block:: python

  add_fun = mx.import_function("add.mlxfn")

  out, = add_fun(mx.array(1.0), mx.array(2.0))
  # Prints: array(3, dtype=float32)
  print(out)

  out, = add_fun(mx.array(1.0), mx.array(3.0))
  # Prints: array(4, dtype=float32)
  print(out)

  # Raises an exception
  add_fun(mx.array(1), mx.array(3.0))

  # Raises an exception
  add_fun(mx.array([1.0, 2.0]), mx.array(3.0))

Notice the third and fourth calls to ``add_fun`` raise exceptions because the
shapes and types of the inputs are different than the shapes and types of the
example inputs we exported the function with.

Also notice that even though the original ``fun`` returns a single output
array, the imported function always returns a tuple of one or more arrays.

The inputs to :func:`export_function` and to an imported function can be
specified as variable positional arguments or as a tuple of arrays:

.. code-block:: python

  def fun(x, y):
    return x + y

  x = mx.array(1.0)
  y = mx.array(1.0)
   
  # Both arguments to fun are positional
  mx.export_function("add.mlxfn", fun, x, y)

  # Same as above
  mx.export_function("add.mlxfn", fun, (x, y))

  imported_fun = mx.import_function("add.mlxfn")

  # Ok
  out, = imported_fun(x, y)

  # Also ok
  out, = imported_fun((x, y))

You can pass example inputs to functions as positional or keyword arguments. If
you use keyword arguments to export the function, then you have to use the same
keyword arguments when calling the imported function.

.. code-block:: python

  def fun(x, y):
    return x + y

  # One argument to fun is positional, the other is a kwarg
  mx.export_function("add.mlxfn", fun, x, y=y)

  imported_fun = mx.import_function("add.mlxfn")

  # Ok
  out, = imported_fun(x, y=y)

  # Also ok
  out, = imported_fun((x,), {"y": y})

  # Raises since the keyword argument is missing
  out, = imported_fun(x, y)

  # Raises since the keyword argument has the wrong key
  out, = imported_fun(x, z=y)


Exporting Modules
-----------------

An :obj:`mlx.nn.Module` can be exported with or without the parameters included
in the exported function. Here's an example:

.. code-block:: python

   model = nn.Linear(4, 4)
   mx.eval(model.parameters())

   def call(x):
      return model(x)

   mx.export_function("model.mlxfn", call, mx.zeros(4))

In the above example, the :obj:`mlx.nn.Linear` module is exported. Its
parameters are also saved to the ``model.mlxfn`` file.

.. note::

   For enclosed arrays inside an exported function, be extra careful to ensure
   they are evaluated. The computation graph that gets exported will include
   the computation that produces enclosed inputs.
  
   If the above example was missing ``mx.eval(model.parameters()``, the
   exported function would include the random initialization of the
   :obj:`mlx.nn.Module` parameters.

If you only want to export the ``Module.__call__`` function without the
parameters, pass them as inputs to the ``call`` wrapper:

.. code-block:: python

   model = nn.Linear(4, 4)
   mx.eval(model.parameters())

   def call(x, **params):
     # Set the model's parameters to the input parameters
     model.update(tree_unflatten(list(params.items())))
     return model(x)
 
   params = dict(tree_flatten(model.parameters()))
   mx.export_function("model.mlxfn", call, (mx.zeros(4),), params)


Shapeless Exports
-----------------

Just like :func:`compile`, functions can also be exported for dynamically shaped
inputs. Pass ``shapeless=True`` to :func:`export_function` or :func:`exporter`
to export a function which can be used for inputs with variable shapes:

.. code-block:: python

  mx.export_function("fun.mlxfn", mx.abs, mx.array(0.0), shapeless=True)
  imported_abs = mx.import_function("fun.mlxfn")

  # Ok
  out, = imported_abs(mx.array(-1.0))
  
  # Also ok 
  out, = imported_abs(mx.array([-1.0, -2.0]))

With ``shapeless=False`` (which is the default), the second call to
``imported_abs`` would raise an exception with a shape mismatch.

Shapeless exporting works the same as shapeless compilation and should be
used carefully. See the :ref:`documentation on shapeless compilation
<shapeless_compile>` for more information.

Exporting Multiple Traces
-------------------------

In some cases, functions build different computation graphs for different
input arguments. A simple way to manage this is to export to a new file with
each set of inputs. This is a fine option in many cases. But it can be
suboptimal if the exported functions have a large amount of duplicate constant
data (for example the parameters of a :obj:`mlx.nn.Module`).

The export API in MLX lets you export multiple traces of the same function to
a single file by creating an exporting context manager with :func:`exporter`:

.. code-block:: python

  def fun(x, y=None):
      constant = mx.array(3.0)
      if y is not None:
        x += y 
      return x + constant

  with mx.exporter("fun.mlxfn", fun) as exporter:
      exporter(mx.array(1.0))
      exporter(mx.array(1.0), y=mx.array(0.0))

  imported_function = mx.import_function("fun.mlxfn")

  # Call the function with y=None
  out, = imported_function(mx.array(1.0))
  print(out)

  # Call the function with y specified
  out, = imported_function(mx.array(1.0), y=mx.array(1.0))
  print(out)

In the above example the function constant data, (i.e. ``constant``), is only
saved once. 

Transformations with Imported Functions
---------------------------------------

Function transformations like :func:`grad`, :func:`vmap`, and :func:`compile` work
on imported functions just like regular Python functions:

.. code-block:: python

  def fun(x):
      return mx.sin(x)

  x = mx.array(0.0)
  mx.export_function("sine.mlxfn", fun, x)

  imported_fun = mx.import_function("sine.mlxfn")

  # Take the derivative of the imported function
  dfdx = mx.grad(lambda x: imported_fun(x)[0])
  # Prints: array(1, dtype=float32)
  print(dfdx(x))

  # Compile the imported function 
  mx.compile(imported_fun)
  # Prints: array(0, dtype=float32)
  print(compiled_fun(x)[0])


Importing Functions in C++
--------------------------

Importing and running functions in C++ is basically the same as importing and
running them in Python. First, follow the :ref:`instructions <mlx_in_cpp>` to
setup a simple C++ project that uses MLX as a library.

Next, export a simple function from Python:

.. code-block:: python

  def fun(x, y):
      return mx.exp(x + y)

  x = mx.array(1.0)
  y = mx.array(1.0)
  mx.export_function("fun.mlxfn", fun, x, y)


Import and run the function in C++ with only a few lines of code:

.. code-block:: c++

  auto fun = mx::import_function("fun.mlxfn");

  auto inputs = {mx::array(1.0), mx::array(1.0)};
  auto outputs = fun(inputs);

  // Prints: array(2, dtype=float32)
  std::cout << outputs[0] << std::endl;

Imported functions can be transformed in C++ just like in Python. Use 
``std::vector<mx::array>`` for positional arguments and ``std::map<std::string,
mx::array>`` for keyword arguments when calling imported functions in C++.

More Examples
-------------

Here are a few more complete examples exporting more complex functions from
Python and importing and running them in C++:

* `Inference and training a multi-layer perceptron <https://github.com/ml-explore/mlx/tree/main/examples/export>`_
