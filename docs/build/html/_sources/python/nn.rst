.. _nn:

.. currentmodule:: mlx.nn

Neural Networks
===============

Writing arbitrarily complex neural networks in MLX can be done using only
:class:`mlx.core.array` and :meth:`mlx.core.value_and_grad`. However, this requires the
user to write again and again the same simple neural network operations as well
as handle all the parameter state and initialization manually and explicitly.

The module :mod:`mlx.nn` solves this problem by providing an intuitive way of
composing neural network layers, initializing their parameters, freezing them
for finetuning and more.

Quick Start with Neural Networks
---------------------------------

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int):
            super().__init__()

            self.layers = [
                nn.Linear(in_dims, 128),
                nn.Linear(128, 128),
                nn.Linear(128, out_dims),
            ]

        def __call__(self, x):
            for i, l in enumerate(self.layers):
                x = mx.maximum(x, 0) if i > 0 else x
                x = l(x)
            return x

    # The model is created with all its parameters but nothing is initialized
    # yet because MLX is lazily evaluated
    mlp = MLP(2, 10)

    # We can access its parameters by calling mlp.parameters()
    params = mlp.parameters()
    print(params["layers"][0]["weight"].shape)

    # Printing a parameter will cause it to be evaluated and thus initialized
    print(params["layers"][0])

    # We can also force evaluate all parameters to initialize the model
    mx.eval(mlp.parameters())

    # A simple loss function.
    # NOTE: It doesn't matter how it uses the mlp model. It currently captures
    #       it from the local scope. It could be a positional argument or a
    #       keyword argument.
    def l2_loss(x, y):
        y_hat = mlp(x)
        return (y_hat - y).square().mean()

    # Calling `nn.value_and_grad` instead of `mx.value_and_grad` returns the
    # gradient with respect to `mlp.trainable_parameters()`
    loss_and_grad = nn.value_and_grad(mlp, l2_loss)

.. _module_class:

The Module Class
----------------

The workhorse of any neural network library is the :class:`Module` class. In
MLX the :class:`Module` class is a container of :class:`mlx.core.array` or
:class:`Module` instances. Its main function is to provide a way to
recursively **access** and **update** its parameters and those of its
submodules.

Parameters
^^^^^^^^^^

A parameter of a module is any public member of type :class:`mlx.core.array` (its
name should not start with ``_``). It can be arbitrarily nested in other
:class:`Module` instances or lists and dictionaries.

:meth:`Module.parameters` can be used to extract a nested dictionary with all
the parameters of a module and its submodules.

A :class:`Module` can also keep track of "frozen" parameters. See the
:meth:`Module.freeze` method for more details. :meth:`mlx.nn.value_and_grad`
the gradients returned will be with respect to these trainable parameters.


Updating the Parameters
^^^^^^^^^^^^^^^^^^^^^^^

MLX modules allow accessing and updating individual parameters. However, most
times we need to update large subsets of a module's parameters. This action is
performed by :meth:`Module.update`.


Inspecting Modules
^^^^^^^^^^^^^^^^^^

The simplest way to see the model architecture is to print it. Following along with
the above example, you can print the ``MLP`` with:

.. code-block:: python

  print(mlp)

This will display:

.. code-block:: shell

  MLP(
    (layers.0): Linear(input_dims=2, output_dims=128, bias=True)
    (layers.1): Linear(input_dims=128, output_dims=128, bias=True)
    (layers.2): Linear(input_dims=128, output_dims=10, bias=True)
  )

To get more detailed information on the arrays in a :class:`Module` you can use
:func:`mlx.utils.tree_map` on the parameters. For example, to see the shapes of
all the parameters in a :class:`Module` do:

.. code-block:: python

   from mlx.utils import tree_map
   shapes = tree_map(lambda p: p.shape, mlp.parameters())

As another example, you can count the number of parameters in a :class:`Module`
with:

.. code-block:: python

   from mlx.utils import tree_flatten
   num_params = sum(v.size for _, v in tree_flatten(mlp.parameters()))


Value and Grad
--------------

Using a :class:`Module` does not preclude using MLX's high order function
transformations (:meth:`mlx.core.value_and_grad`, :meth:`mlx.core.grad`, etc.). However,
these function transformations assume pure functions, namely the parameters
should be passed as an argument to the function being transformed.

There is an easy pattern to achieve that with MLX modules

.. code-block:: python

    model = ...

    def f(params, other_inputs):
        model.update(params)  # <---- Necessary to make the model use the passed parameters
        return model(other_inputs)

    f(model.trainable_parameters(), mx.zeros((10,)))

However, :meth:`mlx.nn.value_and_grad` provides precisely this pattern and only
computes the gradients with respect to the trainable parameters of the model.

In detail:

- it wraps the passed function with a function that calls :meth:`Module.update`
  to make sure the model is using the provided parameters.
- it calls :meth:`mlx.core.value_and_grad` to transform the function into a function
  that also computes the gradients with respect to the passed parameters.
- it wraps the returned function with a function that passes the trainable
  parameters as the first argument to the function returned by
  :meth:`mlx.core.value_and_grad`

.. autosummary::
   :toctree: _autosummary

   value_and_grad
   quantize

.. toctree::

   nn/module
   nn/layers
   nn/functions
   nn/losses
   nn/init
