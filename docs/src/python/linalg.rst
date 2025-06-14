.. _linalg:

Linear Algebra
==============

.. currentmodule:: mlx.core.linalg

MLX provides a comprehensive set of linear algebra operations with GPU acceleration
on Apple Silicon. Many operations, including SVD, are optimized for Metal GPU execution
to provide significant performance improvements over CPU-only implementations.

.. autosummary::
   :toctree: _autosummary

    inv
    tri_inv
    norm
    cholesky
    cholesky_inv
    cross
    qr
    svd
    eigvals
    eig
    eigvalsh
    eigh
    lu
    lu_factor
    pinv
    solve
    solve_triangular
