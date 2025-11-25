# ABOUTME: Initializes the MLX Python namespace for convenience imports.
# ABOUTME: Extends package discovery across editable install paths.

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
