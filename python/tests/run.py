import os
import sys

# Use regular fp32 precision for tests
os.environ["MLX_ENABLE_TF32"] = "0"

# Do not abort on cache thrashing
os.environ["MLX_ENABLE_CACHE_THRASHING_CHECK"] = "0"

__unittest = True

import mlx_tests

if __name__ == "__main__":
    # Run all tests by default.
    dirname = os.path.dirname(os.path.realpath(__file__))
    argv = [sys.argv[0], "discover", dirname, *sys.argv[1:]]
    mlx_tests.MLXTestRunner(argv=argv, module=None)
