"""
MLX: An array framework for Apple silicon.

MLX is an array framework for machine learning on Apple silicon, brought
to you by the Apple machine learning research team.

It features:
- A familiar API (similar to NumPy and PyTorch).
- Composable function transformations (differentiation, vectorization, etc.).
- Lazy computation and dynamic graph construction.
- Multi-device support (CPU and GPU).
"""

from importlib.metadata import PackageNotFoundError, version

from . import nn, optimizers, utils
from .core import *

try:
    __version__ = version("mlx")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = [
    "nn",
    "optimizers",
    "utils",
    "core",
    "__version__",
]

import mlx.core as _core

__all__ += [k for k in dir(_core) if not k.startswith("_")]
