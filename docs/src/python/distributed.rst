.. _distributed:

.. currentmodule:: mlx.core.distributed

Distributed Communication
==========================

MLX provides a distributed communication package using MPI. The MPI library is
loaded at runtime; if MPI is available then distributed communication is also
made available.

.. autosummary::
   :toctree: _autosummary

    Group
    is_available
    init
    all_max
    all_min
    all_sum
    all_gather
    send
    recv
    recv_like
    sum_scatter
