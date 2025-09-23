.. _distributions:

=================================
Distributions (`mlx.distributions`)
=================================

The ``mlx.distributions`` package contains parameterizable probability
distributions and sampling functions.

The design is inspired by ``torch.distributions`` but is adapted for the
functional and JIT-first nature of MLX. Key features include:

- **Differentiable Sampling**: Distributions that support it use the
  reparameterization trick, making sampling a differentiable operation.
- **Composability**: Complex distributions can be built from simpler ones
  using transforms and other utilities.

.. toctree::
   :maxdepth: 2
   :caption: Distributions:

   distributions
   transforms
   composition