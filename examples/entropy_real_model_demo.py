#!/usr/bin/env python3
"""
Entropy-Coded Quantization: Real Model Compression Demo

This example demonstrates entropy coding compression on real quantized models.

IMPORTANT FINDING:
- Per-TENSOR quantization (single scale): ~2.7 bits entropy -> 1.47x compression
- Per-GROUP quantization (mlx-lm default): ~3.7 bits entropy -> 1.07x compression

The compression ratio depends heavily on the quantization granularity.

Usage:
    python entropy_real_model_demo.py

Requirements:
    - mlx-lm: pip install mlx-lm
    - numpy
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Check for mlx-lm
try:
    from mlx_lm import load
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False
    print("[ERROR] mlx-lm not found. Install with: pip install mlx-lm")


def unpack_4bit_weights(packed: np.ndarray, bits: int = 4) -> np.ndarray:
    """Unpack MLX 4-bit packed weights to individual indices."""
    out_features = packed.shape[0]
    packed_cols = packed.shape[1]
    
    if bits == 4:
        # MLX packs 8 x 4-bit values per uint32
        in_features = packed_cols * 8
        indices = np.zeros((out_features, in_features), dtype=np.uint8)
        for i in range(8):
            shift = i * 4
            indices[:, i::8] = (packed >> shift) & 0x0F
        return indices
    else:
        raise ValueError(f"Only 4-bit supported, got {bits}")


def compute_entropy(indices: np.ndarray) -> float:
    """Compute Shannon entropy in bits per symbol."""
    counts = np.bincount(indices.flatten(), minlength=16)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def analyze_quantized_weights(packed_weight: mx.array, bits: int = 4) -> dict:
    """Analyze the distribution of 4-bit quantized weights."""
    packed_np = np.array(packed_weight)
    indices = unpack_4bit_weights(packed_np, bits)
    all_values = indices.flatten()
    
    counts = np.bincount(all_values, minlength=16)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    compression_ratio = 4.0 / entropy if entropy > 0 else 1.0
    
    return {
        "counts": counts,
        "probs": probs,
        "entropy": entropy,
        "compression_ratio": compression_ratio,
        "total_symbols": len(all_values),
    }


def get_model_quantized_layers(model) -> list:
    """Extract all QuantizedLinear layers from model."""
    from mlx.nn.layers.quantized import QuantizedLinear
    
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            layers.append({
                "name": name,
                "layer": module,
                "weight": module.weight,
                "scales": module.scales,
                "biases": getattr(module, 'biases', None),
                "group_size": getattr(module, 'group_size', 64),
                "bits": getattr(module, 'bits', 4),
            })
    
    return layers


def simulate_per_tensor_quantization(model) -> float:
    """
    Simulate per-tensor quantization entropy on a model.
    
    This shows what entropy would be if we used per-tensor instead of per-group.
    """
    from mlx.nn.layers.quantized import QuantizedLinear
    
    total_entropy = 0
    total_symbols = 0
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            # Dequantize the weights
            packed = np.array(module.weight)
            scales = np.array(module.scales)
            biases = np.array(module.biases) if module.biases is not None else None
            
            # Unpack
            indices = unpack_4bit_weights(packed, module.bits)
            
            # Dequantize per-group
            group_size = module.group_size
            rows, cols = indices.shape
            weights = np.zeros((rows, cols), dtype=np.float32)
            
            n_groups = scales.shape[1]
            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, cols)
                if biases is not None:
                    weights[:, start:end] = indices[:, start:end] * scales[:, g:g+1] + biases[:, g:g+1]
                else:
                    weights[:, start:end] = indices[:, start:end] * scales[:, g:g+1]
            
            # Re-quantize per-tensor
            w_min = weights.min()
            w_max = weights.max()
            scale = (w_max - w_min) / 15 if w_max != w_min else 1e-8
            new_indices = np.clip(np.round((weights - w_min) / scale), 0, 15).astype(np.uint8)
            
            entropy = compute_entropy(new_indices)
            total_entropy += entropy * new_indices.size
            total_symbols += new_indices.size
    
    return total_entropy / total_symbols if total_symbols > 0 else 0


def main():
    if not HAS_MLX_LM:
        print("\nPlease install mlx-lm: pip install mlx-lm")
        return
    
    print("=" * 70)
    print("Entropy-Coded Quantization: Real Model Demo")
    print("=" * 70)
    
    # Load a real 4-bit quantized model from mlx-community
    model_name = "mlx-community/SmolLM-135M-4bit"
    
    print(f"\n[1] Loading {model_name}...")
    print("    (Downloads ~80MB on first run)")
    
    model, tokenizer = load(model_name)
    mx.eval(model.parameters())
    
    print("    Done.")
    
    # =========================================================================
    # Analyze weight distributions
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2] ANALYZING PER-GROUP QUANTIZED WEIGHTS (mlx-lm default)")
    print("=" * 70)
    
    layers = get_model_quantized_layers(model)
    print(f"\n    Found {len(layers)} QuantizedLinear layers")
    
    if not layers:
        print("    [WARN] No QuantizedLinear layers found.")
        return
    
    # Analyze all layers
    total_entropy = 0
    total_symbols = 0
    
    for info in layers:
        stats = analyze_quantized_weights(info["weight"], info["bits"])
        total_entropy += stats["entropy"] * stats["total_symbols"]
        total_symbols += stats["total_symbols"]
    
    avg_entropy_group = total_entropy / total_symbols if total_symbols > 0 else 0
    compression_group = 4.0 / avg_entropy_group if avg_entropy_group > 0 else 1.0
    
    print(f"\n    Per-group quantization (group_size=64):")
    print(f"    Average entropy:     {avg_entropy_group:.3f} bits")
    print(f"    Compression ratio:   {compression_group:.2f}x over 4-bit")
    
    # =========================================================================
    # Simulate per-tensor quantization
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] SIMULATING PER-TENSOR QUANTIZATION")
    print("=" * 70)
    
    print("\n    Re-quantizing weights with per-tensor scale...")
    avg_entropy_tensor = simulate_per_tensor_quantization(model)
    compression_tensor = 4.0 / avg_entropy_tensor if avg_entropy_tensor > 0 else 1.0
    
    print(f"\n    Per-tensor quantization:")
    print(f"    Average entropy:     {avg_entropy_tensor:.3f} bits")
    print(f"    Compression ratio:   {compression_tensor:.2f}x over 4-bit")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Model: {model_name}
    
    | Quantization Method     | Entropy    | Compression |
    |-------------------------|------------|-------------|
    | Per-group (mlx-lm)      | {avg_entropy_group:.2f} bits  | {compression_group:.2f}x        |
    | Per-tensor (simulated)  | {avg_entropy_tensor:.2f} bits  | {compression_tensor:.2f}x        |
    
    KEY INSIGHT:
    Entropy coding compression depends on quantization granularity:
    
    - Per-GROUP quantization (group_size=64) distributes values uniformly
      within each group, resulting in ~3.7 bits entropy and only ~1.07x
      compression.
    
    - Per-TENSOR quantization (single scale) preserves the Gaussian
      distribution, concentrating values in middle bins (7-8), resulting
      in ~2.7 bits entropy and ~1.47x compression.
    
    IMPLICATIONS:
    1. For mlx-lm models (per-group): Entropy coding provides ~7% savings
    2. For per-tensor models: Entropy coding provides ~47% savings
    3. Best use case: Disk storage compression (decode once at load time)
    """)


if __name__ == "__main__":
    main()
