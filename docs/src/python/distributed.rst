.. _distributed:

.. currentmodule:: mlx.core.distributed

Distributed Communication
==========================

MLX supports distributed communication with the MPI, Ring, JACCL, and NCCL
backends. See :ref:`usage_distributed` for backend availability and setup.

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
