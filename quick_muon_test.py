#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'python')

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import Muon
    
    print("✅ Import successful")
    
    # Test basic instantiation
    opt = Muon(learning_rate=0.01)
    print("✅ Instantiation successful")
    
    # Test with a simple linear layer
    layer = nn.Linear(3, 2)
    x = mx.random.normal([1, 3])
    y = mx.random.normal([1, 2])
    
    def loss_fn(layer, x, y):
        return mx.mean((layer(x) - y) ** 2)
    
    loss_and_grad_fn = nn.value_and_grad(layer, loss_fn)
    loss, grads = loss_and_grad_fn(layer, x, y)
    
    opt.update(layer, grads)
    print("✅ Basic optimization successful")
    
    print("🎉 All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
