# Copyright © 2026 Apple Inc.


import mlx.core as mx


def pytest_configure(config):
    # Use regular fp32 precision for tests
    mx.config.update("MLX_ENABLE_TF32", 0)
    # Do not abort on cache thrashing
    mx.config.update("MLX_ENABLE_CACHE_THRASHING_CHECK", 0)


def pytest_unconfigure(config):
    mx.clear_streams()
