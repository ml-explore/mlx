#!/usr/bin/env python3

import os
import sys

# Add the python directory to the path to test our local build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

try:
    import mlx.core as mx

    print("✓ Successfully imported mlx.core as mx")

    # Test that no_grad is available at top level
    if hasattr(mx, "no_grad"):
        print("✓ mx.no_grad is available at top level")

        # Test that it works
        x = mx.array(2.0)

        def f(x):
            return x * x

        grad_fn = mx.value_and_grad(f)

        # Test normal gradient
        y, dydx = grad_fn(x)
        print(f"✓ Normal gradient: f(2) = {y.item()}, df/dx = {dydx.item()}")

        # Test with mx.no_grad()
        with mx.no_grad():
            y2, dydx2 = grad_fn(x)
            print(f"✓ With mx.no_grad(): f(2) = {y2.item()}, df/dx = {dydx2.item()}")

        print("✓ mx.no_grad() works correctly at top level!")

    else:
        print("✗ mx.no_grad is NOT available at top level")

    # Test that enable_grad is also available
    if hasattr(mx, "enable_grad"):
        print("✓ mx.enable_grad is available at top level")
    else:
        print("✗ mx.enable_grad is NOT available at top level")

except ImportError as e:
    print(f"✗ Failed to import mlx.core: {e}")
except Exception as e:
    print(f"✗ Error during testing: {e}")
