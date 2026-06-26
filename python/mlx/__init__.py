# Copyright © 2026 Apple Inc.
"""MLX: A framework for machine learning on Apple silicon.

This module makes ``mlx`` a regular package (previously a namespace
package) so that pure-Python transforms can be registered onto the
``mlx.core`` C extension *after* it has been fully initialized.

Registering a pure-Python attribute on ``mlx.core`` from inside the
C extension's ``NB_MODULE`` (e.g. via ``_reprlib_fix``) is unsafe because
that hook runs before ``mlx.core`` is in ``sys.modules``, so an
``import mlx.core`` from there re-enters ``NB_MODULE`` and fatally
double-registers nanobind enumerations. Importing ``mlx.core`` from here
instead guarantees the C extension is fully built before we patch it.
"""

# Import the C extension FIRST so it is fully initialized and in
# ``sys.modules`` before ``_while_loop`` is imported (``_while_loop`` does
# ``import mlx.core`` at module top and relies on finding it already loaded).
# When a user does ``import mlx.core as mx``, Python imports the parent package
# ``mlx`` (running this file), which loads and fully initializes ``mlx.core``;
# the subsequent ``import mlx.core`` then finds it already in ``sys.modules``.
# `isort: off` preserves this required order (isort would otherwise sort
# ``_while_loop`` ahead of ``core`` alphabetically).
# isort: off
from . import core as _core  # noqa: F401  (ensures C extension is loaded)
from . import _while_loop as _wl  # noqa: F401

# isort: on

_core.while_loop = _wl.while_loop

del _core, _wl
