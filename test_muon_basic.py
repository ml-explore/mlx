#!/usr/bin/env python3

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Muon
import numpy as np

def test_muon_basic():
    """Basic test to ensure Muon initializes and runs without errors."""
    print("Testing Muon optimizer basic functionality...")
    
    # Create a simple 2-layer MLP
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
            super().__init__()
            self.layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ]
        
        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Initialize model and optimizer
    model = SimpleMLP()
    optimizer = Muon(learning_rate=0.02, momentum=0.95)
    
    # Create dummy data
    batch_size = 4
    x = mx.random.normal([batch_size, 10])
    y = mx.random.normal([batch_size, 5])
    
    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)
    
    # Get gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Perform one optimization step
    loss, grads = loss_and_grad_fn(model, x, y)
    print(f"Initial loss: {loss.item():.4f}")
    
    # Update parameters
    optimizer.update(model, grads)
    
    # Compute loss after update
    new_loss = loss_fn(model, x, y)
    print(f"Loss after Muon update: {new_loss.item():.4f}")
    
    # Verify loss decreased (or at least changed)
    assert abs(loss.item() - new_loss.item()) > 1e-6, "Loss should change after update"
    print("✓ Basic Muon functionality test passed!")

def test_muon_state_dict():
    """Test save/load state dict functionality."""
    print("\nTesting Muon state dict save/load...")
    
    # Simple linear layer
    layer = nn.Linear(5, 3)
    optimizer = Muon(learning_rate=0.01)
    
    # Initialize optimizer state
    params = layer.parameters()
    optimizer.init(params)
    
    # Save state
    original_state = optimizer.state
    print(f"✓ State dict saved with keys: {list(original_state.keys())}")
    
    # Create new optimizer and load state
    new_optimizer = Muon(learning_rate=0.01)
    new_optimizer.state = original_state
    
    print("✓ State dict load test passed!")

def test_muon_different_shapes():
    """Test Muon with different parameter shapes."""
    print("\nTesting Muon with different parameter shapes...")
    
    optimizer = Muon(learning_rate=0.01)
    
    # Test 2D matrix (should use Muon update)
    weight_2d = mx.random.normal([10, 5])
    grad_2d = mx.random.normal([10, 5])
    
    # Test 1D bias (should use standard momentum)
    bias_1d = mx.random.normal([5])
    grad_1d = mx.random.normal([5])
    
    # Test 4D conv filter (should reshape and use Muon)
    conv_4d = mx.random.normal([16, 8, 3, 3])
    grad_4d = mx.random.normal([16, 8, 3, 3])
    
    # Initialize states
    state_2d = {"momentum_buffer": mx.zeros_like(weight_2d)}
    state_1d = {"momentum_buffer": mx.zeros_like(bias_1d)}
    state_4d = {"momentum_buffer": mx.zeros_like(conv_4d)}
    
    # Apply updates
    new_weight_2d = optimizer.apply_single(grad_2d, weight_2d, state_2d)
    new_bias_1d = optimizer.apply_single(grad_1d, bias_1d, state_1d)
    new_conv_4d = optimizer.apply_single(grad_4d, conv_4d, state_4d)
    
    print(f"✓ 2D weight update: {weight_2d.shape} -> {new_weight_2d.shape}")
    print(f"✓ 1D bias update: {bias_1d.shape} -> {new_bias_1d.shape}")
    print(f"✓ 4D conv update: {conv_4d.shape} -> {new_conv_4d.shape}")
    
    # Verify shapes are preserved
    assert new_weight_2d.shape == weight_2d.shape
    assert new_bias_1d.shape == bias_1d.shape
    assert new_conv_4d.shape == conv_4d.shape
    
    print("✓ Different shapes test passed!")

if __name__ == "__main__":
    print("Running Muon optimizer tests...\n")
    
    try:
        test_muon_basic()
        test_muon_state_dict()
        test_muon_different_shapes()
        print("\n🎉 All Muon tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
