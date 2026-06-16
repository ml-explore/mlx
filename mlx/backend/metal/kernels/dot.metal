// Copyright © 2026 Apple Inc.

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void dot_product(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]) {
  constexpr int ITEMS_PER_THREAD = 32;

  int start = gid * ITEMS_PER_THREAD;

  float c0 = 0.0f;
  float c1 = 0.0f;
  float c2 = 0.0f;
  float c3 = 0.0f;

  // Fast path: no per-element branches.
  if (start + ITEMS_PER_THREAD <= n) {
    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < ITEMS_PER_THREAD; i += 4) {
      c0 += float(a[start + i + 0]) * float(b[start + i + 0]);
      c1 += float(a[start + i + 1]) * float(b[start + i + 1]);
      c2 += float(a[start + i + 2]) * float(b[start + i + 2]);
      c3 += float(a[start + i + 3]) * float(b[start + i + 3]);
    }
  } else {
    // Tail path only for the last few threads.
    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      int idx = start + i;
      if (idx < n) {
        float v = float(a[idx]) * float(b[idx]);
        switch (i & 3) {
          case 0:
            c0 += v;
            break;
          case 1:
            c1 += v;
            break;
          case 2:
            c2 += v;
            break;
          default:
            c3 += v;
            break;
        }
      }
    }
  }

  threadgroup float smem[16];

  float c = c0 + c1 + c2 + c3;
  c = simd_sum(c);

  if (lane == 0) {
    smem[simd_id] = c;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 16) {
    c = smem[tid];
    c = simd_sum(c);
    if (tid == 0) {
      output[tg_id] = c;
    }
  }
}

template <typename T>
[[kernel]] void dot_reduce(
    const device float* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    const constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]) {
  float c = gid < uint(n) ? float(input[gid]) : 0.0f;

  threadgroup float smem[16];

  c = simd_sum(c);
  if (lane == 0) {
    smem[simd_id] = c;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 16) {
    c = smem[tid];
    c = simd_sum(c);
    if (tid == 0) {
      output[tg_id] = T(c);
    }
  }
}

#define instantiate_dot_product_kernel(name, itype) \
  instantiate_kernel("dot_product_" #name, dot_product, itype)

#define instantiate_dot_reduce_kernel(name, otype) \
  instantiate_kernel("dot_reduce_" #name, dot_reduce, otype)

instantiate_dot_product_kernel(float32, float);
instantiate_dot_product_kernel(float16, half);
instantiate_dot_product_kernel(bfloat16, bfloat16_t);
instantiate_dot_reduce_kernel(float32, float);
instantiate_dot_reduce_kernel(float16, half);
instantiate_dot_reduce_kernel(bfloat16, bfloat16_t);