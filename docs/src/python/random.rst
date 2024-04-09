.. _random:

Random
======

Random sampling functions in MLX use an implicit global PRNG state by default.
However, all function take an optional ``key`` keyword argument for when more
fine-grained control or explicit state management is needed.

For example, you can generate random numbers with:

.. code-block:: python

  for _ in range(3):
    print(mx.random.uniform())

which will print a sequence of unique pseudo random numbers. Alternatively you
can explicitly set the key:

.. code-block:: python

  key = mx.random.key(0)
  for _ in range(3):
    print(mx.random.uniform(key=key))

which will yield the same pseudo random number at each iteration.

Following `JAX's PRNG design <https://jax.readthedocs.io/en/latest/jep/263-prng.html>`_
we use a splittable version of Threefry, which is a counter-based PRNG.

.. currentmodule:: mlx.core.random

.. autosummary:: 
  :toctree: _autosummary

   bernoulli
   categorical
   gumbel
   key
   normal
   multivariate_normal
   randint
   seed
   split
   truncated_normal
   uniform
