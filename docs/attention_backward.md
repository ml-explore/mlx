# Scaled Dot-Product Attention Backward Pass (VJP)

## Overview

MLX provides two backward (VJP) implementations for `mx.fast.scaled_dot_product_attention`:

- **Unfused backward**: Decomposes attention into separate matmul, softmax, and matmul operations. Each operation uses Apple's NAX-optimized matmul kernels which achieve high throughput on Apple Silicon (up to 10.7 TFLOPS with large tiles).

- **Fused backward (STEEL VJP)**: A Flash Attention-style backward pass that recomputes the attention matrix tile-by-tile, avoiding materialization of the full O(L²) attention matrix. Uses two Metal kernels (`steel_attention_vjp_dq` for dQ gradients, `steel_attention_vjp_dkv` for dK/dV gradients).

## Performance vs Memory Tradeoff

On Apple Silicon, the unfused path is **faster** for typical sequence lengths because NAX-optimized matmuls use large tiles (64×64+) that achieve much higher MMA utilization than the fused kernel's 32×32 tiles. The fused kernel also performs 1.75× more FLOPs due to recomputing the attention matrix in both the dQ and dKV kernels.

However, the fused path uses **dramatically less memory** by avoiding the O(L²) attention matrix intermediate:

| L | Unfused Peak | Fused Peak | Memory Savings |
|---|-------------|------------|----------------|
| 512 | ~8 MB | ~3 MB | 67% |
| 1024 | ~29 MB | ~4 MB | 85% |
| 2048 | ~110 MB | ~8 MB | 93% |
| 4096 | ~428 MB | ~19 MB | 96% |

For long sequences (L ≥ 8192), the attention matrix can exceed available GPU memory, making the fused path essential.

This tradeoff is inherent to Flash Attention on Apple Silicon. See [Draw Things' Metal FlashAttention 2.0 discussion](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c) for more context on backward pass challenges specific to Apple GPUs.

## Dispatch Policy

The backward pass implementation is selected automatically, with user control via environment variables.

### `MLX_SDPA_VJP_MODE`

Controls which backward implementation is used. Default: `auto`.

| Mode | Behavior |
|------|----------|
| `auto` | Unfused for typical shapes (fast). Fused when sequence length or estimated attention matrix size exceeds thresholds (memory-safe). |
| `unfused` | Always use unfused backward. Fastest on Apple Silicon but may OOM on long sequences. |
| `fused` | Always use fused backward. Slower but memory-efficient. Use for long-context training or when memory is constrained. |

### `MLX_SDPA_VJP_LONG_L_THRESHOLD`

In `auto` mode, use fused backward when query sequence length ≥ this value. Default: `8192`.

```bash
# Lower the threshold to use fused at shorter sequences
export MLX_SDPA_VJP_LONG_L_THRESHOLD=4096
```

### `MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD`

In `auto` mode, use fused backward when the estimated attention matrix size (B × H × L × L × dtype_size) exceeds this many bytes. Default: `1073741824` (1 GB).

```bash
# Trigger fused when attention matrix would exceed 512 MB
export MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD=536870912
```

## Eligibility

Fused backward is only available when all conditions are met:

- Head dimension D = 64
- Data type: float16 or bfloat16
- No attention mask
- No attention sinks
- Query sequence length > 8 (shorter sequences use the vector VJP kernel)

When these conditions are not met, unfused backward is always used regardless of the mode setting.

## Practical Guidance

**For inference / short-context training** (L < 4096): The default `auto` mode uses unfused, which is fastest.

**For long-context training** (L ≥ 8192): The default `auto` mode automatically switches to fused to prevent OOM.

**For memory-constrained environments**: Set `MLX_SDPA_VJP_MODE=fused` to always use the memory-efficient path, accepting ~13% slower backward passes.

**For maximum speed**: Set `MLX_SDPA_VJP_MODE=unfused` to always use NAX-optimized matmuls, but be aware of quadratic memory growth with sequence length.

## Verifying with Xcode GPU Tools

To inspect which path is actually running and verify memory behavior:

```bash
# Capture a GPU trace
MTL_CAPTURE_ENABLED=1 python benchmarks/python/profile_sdpa_vjp.py \
    --capture --mode fused --L 1024

# Open the resulting .gputrace in Xcode and check:
# 1. Per-kernel duration and occupancy
# 2. Threadgroup memory usage (32KB limit on Apple GPUs)
# 3. Buffer allocation sizes (fused avoids the L×L attention buffer)
```

### Reference Documentation

- [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) — GPU resource limits
- [Analyzing Metal memory usage](https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app) — Memory profiling
- [Analyzing your Metal workload](https://developer.apple.com/documentation/xcode/analyzing-your-metal-workload) — GPU performance analysis
- [Metal developer workflows](https://developer.apple.com/documentation/xcode/metal-developer-workflows/) — Debugging and profiling overview
