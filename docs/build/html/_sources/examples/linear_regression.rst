.. _linear_regression:

Linear Regression
-----------------

Let's implement a basic linear regression model as a starting point to
learn MLX. First import the core package and setup some problem metadata:

.. code-block:: python

  import mlx.core as mx

  num_features = 100
  num_examples = 1_000
  num_iters = 10_000  # iterations of SGD
  lr = 0.01  # learning rate for SGD


We'll generate a synthetic dataset by:

1. Sampling the design matrix ``X``.
2. Sampling a ground truth parameter vector ``w_star``.
3. Compute the dependent values ``y`` by adding Gaussian noise to ``X @ w_star``.

.. code-block:: python

  # True parameters
  w_star = mx.random.normal((num_features,))

  # Input examples (design matrix)
  X = mx.random.normal((num_examples, num_features))

  # Noisy labels
  eps = 1e-2 * mx.random.normal((num_examples,))
  y = X @ w_star + eps


We will use SGD to find the optimal weights. To start, define the squared loss
and get the gradient function of the loss with respect to the parameters.

.. code-block:: python

  def loss_fn(w):
      return 0.5 * mx.mean(mx.square(X @ w - y))

  grad_fn = mx.grad(loss_fn)

Start the optimization by initializing the parameters ``w`` randomly. Then
repeatedly update the parameters for ``num_iters`` iterations. 

.. code-block:: python

  w = 1e-2 * mx.random.normal((num_features,))

  for _ in range(num_iters):
      grad = grad_fn(w)
      w = w - lr * grad
      mx.eval(w)

Finally, compute the loss of the learned parameters and verify that they are
close to the ground truth parameters.

.. code-block:: python

  loss = loss_fn(w)
  error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5

  print(
      f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f}, "
  )
  # Should print something close to: Loss 0.00005, |w-w*| = 0.00364

Complete `linear regression
<https://github.com/ml-explore/mlx/tree/main/examples/python/linear_regression.py>`_
and `logistic regression
<https://github.com/ml-explore/mlx/tree/main/examples/python/logistic_regression.py>`_
examples are available in the MLX GitHub repo.
