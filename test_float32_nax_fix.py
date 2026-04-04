#!/usr/bin/env python3
"""
Test script for float32 NAX gather_mm fix (issue #3362)

This validates that the fix in PR #3369 resolves the crash:
  RuntimeError: Unable to load function steel_gather_mm_rhs_nax_nt_float32_float32...

Before fix: Crashes on M5 Max with TF32 enabled
After fix: Should execute successfully
"""

import mlx.core as mx
import numpy as np

def test_float32_gather_mm_nax():
    """Test float32 gather_mm with NAX path (M5 Max + TF32)."""
    
    print("Testing float32 gather_mm (NAX path)...")
    print(f"Device: {mx.default_device()}")
    print(f"GPU available: {mx.is_available(mx.gpu)}")
    
    if not mx.is_available(mx.gpu):
        print("⚠️  GPU not available, skipping test")
        return
    
    # Create test data (MoE-style dimensions)
    a = mx.random.normal(shape=(8, 256, 128), dtype=mx.float32)
    b = mx.random.normal(shape=(4, 128, 256), dtype=mx.float32)
    
    # MoE-style indices (expert selection)
    lhs_indices = mx.array([0, 2, 5, 7], dtype=mx.uint32)
    rhs_indices = mx.array([1, 0, 3, 2], dtype=mx.uint32)
    
    print(f"Input shapes: a={a.shape}, b={b.shape}")
    print(f"Indices: lhs={lhs_indices.tolist()}, rhs={rhs_indices.tolist()}")
    
    # This should NOT crash with:
    # "Unable to load function steel_gather_mm_rhs_nax_nt_float32_float32..."
    try:
        out = mx.gather_mm(a, b, lhs_indices, rhs_indices)
        mx.eval(out)
        
        print(f"✅ gather_mm executed successfully")
        print(f"Output shape: {out.shape}")
        print(f"Output dtype: {out.dtype}")
        
        # Verify correctness
        a_selected = a[lhs_indices.tolist()]
        b_selected = b[rhs_indices.tolist()]
        expected = a_selected @ b_selected
        
        if mx.allclose(out, expected, atol=1e-5):
            print("✅ Output matches reference implementation")
            return True
        else:
            print("❌ Output does not match reference")
            return False
            
    except RuntimeError as e:
        if "Unable to load function" in str(e) and "steel_gather_mm" in str(e):
            print(f"❌ FAILED: Missing float32 NAX kernel")
            print(f"Error: {e}")
            return False
        else:
            raise

if __name__ == "__main__":
    success = test_float32_gather_mm_nax()
    
    if success:
        print("\n🎉 Test passed! Float32 NAX gather_mm works correctly.")
        exit(0)
    else:
        print("\n💥 Test failed! Float32 NAX gather_mm is broken.")
        exit(1)
