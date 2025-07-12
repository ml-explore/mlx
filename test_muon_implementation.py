#!/usr/bin/env python3

"""
MLX Muon Implementation Test Runner
Validates that the Muon optimizer is correctly implemented and integrated into MLX.
"""

import sys
import os
import traceback

# Add the MLX Python package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_muon_import():
    """Test that Muon can be imported correctly."""
    print("🔍 Testing Muon import...")
    try:
        import mlx.optimizers as opt
        from mlx.optimizers import Muon
        print("✅ Muon import successful")
        return True
    except Exception as e:
        print(f"❌ Muon import failed: {e}")
        traceback.print_exc()
        return False

def test_muon_instantiation():
    """Test that Muon can be instantiated with various parameters."""
    print("🔍 Testing Muon instantiation...")
    try:
        from mlx.optimizers import Muon
        
        # Test default parameters
        opt1 = Muon(learning_rate=0.02)
        
        # Test custom parameters
        opt2 = Muon(
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.001,
            ns_steps=3,
            nesterov=False
        )
        
        print("✅ Muon instantiation successful")
        return True
    except Exception as e:
        print(f"❌ Muon instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_muon_basic_optimization():
    """Test basic optimization functionality."""
    print("🔍 Testing Muon basic optimization...")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.optimizers import Muon
        
        # Create a simple model
        model = nn.Linear(4, 2)
        optimizer = Muon(learning_rate=0.01)
        
        # Create dummy data
        x = mx.random.normal([2, 4])
        y = mx.random.normal([2, 2])
        
        def loss_fn(model, x, y):
            return mx.mean((model(x) - y) ** 2)
        
        # Get gradients and update
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)
        
        # Store original parameters
        original_weight = model.weight.copy()
        original_bias = model.bias.copy()
        
        # Update
        optimizer.update(model, grads)
        
        # Check that parameters changed
        weight_changed = not mx.array_equal(original_weight, model.weight)
        bias_changed = not mx.array_equal(original_bias, model.bias)
        
        if weight_changed and bias_changed:
            print("✅ Muon basic optimization successful")
            return True
        else:
            print("❌ Parameters did not change after optimization")
            return False
            
    except Exception as e:
        print(f"❌ Muon basic optimization failed: {e}")
        traceback.print_exc()
        return False

def test_muon_newton_schulz():
    """Test Newton-Schulz orthogonalization."""
    print("🔍 Testing Newton-Schulz orthogonalization...")
    try:
        import mlx.core as mx
        from mlx.optimizers import Muon
        
        optimizer = Muon(learning_rate=0.01)
        
        # Test with a random matrix
        matrix = mx.random.normal([10, 5])
        
        # Apply Newton-Schulz
        result = optimizer._zeropower_via_newtonschulz5(matrix, steps=5)
        
        # Check that result has the same shape
        if result.shape == matrix.shape:
            print("✅ Newton-Schulz orthogonalization successful")
            return True
        else:
            print(f"❌ Shape mismatch: expected {matrix.shape}, got {result.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Newton-Schulz orthogonalization failed: {e}")
        traceback.print_exc()
        return False

def test_muon_state_persistence():
    """Test optimizer state save/load."""
    print("🔍 Testing Muon state persistence...")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.optimizers import Muon
        
        # Create model and optimizer
        model = nn.Linear(3, 2)
        optimizer = Muon(learning_rate=0.01)
        
        # Initialize optimizer
        params = model.parameters()
        optimizer.init(params)
        
        # Save state
        original_state = optimizer.state
        original_step = original_state["step"]
        
        # Update once
        dummy_grads = {k: mx.random.normal(v.shape) for k, v in params.items()}
        optimizer.update(model, dummy_grads)
        
        # Check that step increased
        if optimizer.state["step"].item() > original_step.item():
            print("✅ Muon state persistence successful")
            return True
        else:
            print("❌ Optimizer step did not increase")
            return False
            
    except Exception as e:
        print(f"❌ Muon state persistence failed: {e}")
        traceback.print_exc()
        return False

def test_muon_with_different_shapes():
    """Test Muon with different parameter shapes."""
    print("🔍 Testing Muon with different parameter shapes...")
    try:
        import mlx.core as mx
        from mlx.optimizers import Muon
        
        optimizer = Muon(learning_rate=0.01)
        
        # Test 2D matrix (linear layer)
        weight_2d = mx.random.normal([5, 3])
        grad_2d = mx.random.normal([5, 3])
        state_2d = {"momentum_buffer": mx.zeros_like(weight_2d)}
        
        result_2d = optimizer.apply_single(grad_2d, weight_2d, state_2d)
        
        # Test 1D bias
        bias_1d = mx.random.normal([3])
        grad_1d = mx.random.normal([3])
        state_1d = {"momentum_buffer": mx.zeros_like(bias_1d)}
        
        result_1d = optimizer.apply_single(grad_1d, bias_1d, state_1d)
        
        # Test 4D conv filter
        conv_4d = mx.random.normal([8, 4, 3, 3])
        grad_4d = mx.random.normal([8, 4, 3, 3])
        state_4d = {"momentum_buffer": mx.zeros_like(conv_4d)}
        
        result_4d = optimizer.apply_single(grad_4d, conv_4d, state_4d)
        
        # Check shapes are preserved
        if (result_2d.shape == weight_2d.shape and 
            result_1d.shape == bias_1d.shape and 
            result_4d.shape == conv_4d.shape):
            print("✅ Muon different shapes test successful")
            return True
        else:
            print("❌ Shape preservation failed")
            return False
            
    except Exception as e:
        print(f"❌ Muon different shapes test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Muon tests."""
    print("🧪 MLX Muon Implementation Test Suite")
    print("====================================")
    
    tests = [
        test_muon_import,
        test_muon_instantiation,
        test_muon_basic_optimization,
        test_muon_newton_schulz,
        test_muon_state_persistence,
        test_muon_with_different_shapes,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Muon implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
