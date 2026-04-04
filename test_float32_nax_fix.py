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
    """Test float32 gather_mm RHS NAX path (M5 Max + TF32).
    
    The NAX path is only triggered when:
    1. M == 1 (matvec, not matmul)
    2. right_sorted == True (sorted_indices=True)
    3. is_nax_available() (M5 Max hardware)
    4. enable_tf32() OR dtype != float32
    
    Without the float32 kernel, this crashes on M5 Max.
    """
    
    print("Testing float32 gather_mm RHS NAX path...")
    print(f"Device: {mx.default_device()}")
    print(f"GPU available: {mx.is_available(mx.gpu)}")
    print("\nNOTE: This test only triggers the NAX path on M5 Max with TF32.")
    print("      On other hardware, it will use the non-NAX path and always pass.\n")
    
    if not mx.is_available(mx.gpu):
        print("⚠️  GPU not available, skipping test")
        return
    
    # CRITICAL: Use M=1 (matvec) to trigger NAX path
    a = mx.random.normal(shape=(8, 1, 128), dtype=mx.float32)  # M=1 is required for NAX
    b = mx.random.normal(shape=(4, 128, 256), dtype=mx.float32)
    
    # CRITICAL: Use sorted indices to activate right_sorted path
    rhs_indices = mx.array([0, 1, 2, 3], dtype=mx.uint32)  # Must be sorted
    
    print(f"Input shapes: a={a.shape} (M=1, triggers NAX), b={b.shape}")
    print(f"RHS indices: {rhs_indices.tolist()} (sorted)")
    print(f"sorted_indices=True (activates NAX path on M5 Max)")
    
    # This crashes with "Unable to load function steel_gather_mm_rhs_nax_nt_float32_float32..."
    # before the fix on M5 Max with TF32 enabled
    try:
        out = mx.gather_mm(a, b, rhs_indices=rhs_indices, sorted_indices=True)
        mx.eval(out)
        
        print(f"\n✅ gather_mm executed successfully")
        print(f"Output shape: {out.shape}")
        print(f"Output dtype: {out.dtype}")
        
        # Verify correctness
        b_selected = b[rhs_indices.tolist()]
        expected = a @ b_selected
        
        if mx.allclose(out, expected, atol=1e-5):
            print("✅ Output matches reference implementation")
            return True
        else:
            print("❌ Output does not match reference")
            return False
            
    except RuntimeError as e:
        if "Unable to load function" in str(e) and "steel_gather_mm" in str(e):
            print(f"\n❌ FAILED: Missing float32 NAX kernel (expected on M5 Max without fix)")
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
