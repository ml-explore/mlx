.. _mlp:

Multi-Layer Perceptron
----------------------

In this example we'll learn to use ``mlx.nn`` by implementing a simple
multi-layer perceptron to classify MNIST.

As a first step import the MLX packages we need:

.. code-block:: python

  import mlx.core as mx
  import mlx.nn as nn
  import mlx.optimizers as optim

  import numpy as np


The model is defined as the ``MLP`` class which inherits from
:class:`mlx.nn.Module`. We follow the standard idiom to make a new module:

1. Define an ``__init__`` where the parameters and/or submodules are setup. See
   the :ref:`Module class docs<module_class>` for more information on how
   :class:`mlx.nn.Module` registers parameters.
2. Define a ``__call__`` where the computation is implemented.

.. code-block:: python

  class MLP(nn.Module):
      def __init__(
          self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
      ):
          super().__init__()
          layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
          self.layers = [
              nn.Linear(idim, odim)
              for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
          ]

      def __call__(self, x):
          for l in self.layers[:-1]:
              x = mx.maximum(l(x), 0.0)
          return self.layers[-1](x)


We define the loss function which takes the mean of the per-example cross
entropy loss.  The ``mlx.nn.losses`` sub-package has implementations of some
commonly used loss functions.

.. code-block:: python

  def loss_fn(model, X, y):
      return mx.mean(nn.losses.cross_entropy(model(X), y))

We also need a function to compute the accuracy of the model on the validation
set:

.. code-block:: python

  def eval_fn(model, X, y):
      return mx.mean(mx.argmax(model(X), axis=1) == y)

Next, setup the problem parameters and load the data:

.. code-block:: python

  num_layers = 2
  hidden_dim = 32
  num_classes = 10
  batch_size = 256
  num_epochs = 10
  learning_rate = 1e-1

  # Load the data
  import mnist 
  train_images, train_labels, test_images, test_labels = map(
      mx.array, mnist.mnist()
  )

Since we're using SGD, we need an iterator which shuffles and constructs
minibatches of examples in the training set:

.. code-block:: python

  def batch_iterate(batch_size, X, y):
      perm = mx.array(np.random.permutation(y.size))
      for s in range(0, y.size, batch_size):
          ids = perm[s : s + batch_size]
          yield X[ids], y[ids]


Finally, we put it all together by instantiating the model, the
:class:`mlx.optimizers.SGD` optimizer, and running the training loop:

.. code-block:: python

  # Load the model
  model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
  mx.eval(model.parameters())

  # Get a function which gives the loss and gradient of the
  # loss with respect to the model's trainable parameters
  loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

  # Instantiate the optimizer
  optimizer = optim.SGD(learning_rate=learning_rate)

  for e in range(num_epochs):
      for X, y in batch_iterate(batch_size, train_images, train_labels):
          loss, grads = loss_and_grad_fn(model, X, y)

          # Update the optimizer state and model parameters
          # in a single call
          optimizer.update(model, grads)

          # Force a graph evaluation
          mx.eval(model.parameters(), optimizer.state)

      accuracy = eval_fn(model, test_images, test_labels)
      print(f"Epoch {e}: Test accuracy {accuracy.item():.3f}")


.. note::
  The :func:`mlx.nn.value_and_grad` function is a convenience function to get
  the gradient of a loss with respect to the trainable parameters of a model.
  This should not be confused with :func:`mlx.core.value_and_grad`.

The model should train to a decent accuracy (about 95%) after just a few passes
over the training set. The `full example <https://github.com/ml-explore/mlx-examples/tree/main/mlp>`_
is available in the MLX GitHub repo.
