#pragma once

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <
    typename T,
    const int ITEMS_PER_THREAD,
    const int TG_SIZE,
    const uint SIMD_GROUPS>
[[kernel]] void dot_product(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]) {
  constexpr int VEC = 16 / sizeof(T);
  int start = (tg_id * TG_SIZE + simd_id * 32) * ITEMS_PER_THREAD + lane * VEC;

  float4 c = 0.0f;

  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < ITEMS_PER_THREAD; i += VEC) {
    int idx = start + i * ITEMS_PER_THREAD;
    if (idx + VEC <= n) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < VEC; j += 4) {
        c += float4(*reinterpret_cast<const device metal::vec<T, 4>*>(
                 a + idx + j)) *
            float4(*reinterpret_cast<const device metal::vec<T, 4>*>(
                b + idx + j));
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < VEC; ++j) {
        int nidx = idx + j;
        if (nidx < n) {
          c[j & 3] += float(a[nidx]) * float(b[nidx]);
        }
      }
    }
  }

  threadgroup float smem[SIMD_GROUPS];

  float sum = c[0] + c[1] + c[2] + c[3];
  sum = simd_sum(sum);

  if (lane == 0) {
    smem[simd_id] = sum;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < SIMD_GROUPS) {
    sum = smem[tid];
    sum = simd_sum(sum);
    if (tid == 0) {
      output[tg_id] = sum;
    }
  }
}
