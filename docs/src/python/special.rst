.. _special:

Special Functions
=================

The :mod:`mlx.special` module provides MLX-native implementations of common
special functions, mirroring a subset of :mod:`scipy.special`. Each function is
implemented with pure ``mlx.core`` operations, so results are lazy,
differentiable, and run on the same device as their inputs.

.. currentmodule:: mlx.special

.. autosummary::
  :toctree: _autosummary

   erf
   erfc
   i0
   gammaln
   digamma
