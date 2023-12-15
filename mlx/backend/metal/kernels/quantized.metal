// Copyright © 2023 Apple Inc.

#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

template <typename T, const int BM, const int BN, const int groups, const int width>
[[kernel]] void qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]], 
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  static_assert(BN == SIMD_SIZE, "qmv expects BN to be equal to SIMD_SIZE");

  constexpr int TM = 4;
  constexpr int bitmask = (1 << width) - 1;
  constexpr int el_per_thread = 32 / width;
  constexpr int colgroup = BN * el_per_thread;
  constexpr int groups_per_block = colgroup / groups;
  constexpr int simdgroups_fetching_vec = colgroup / SIMD_SIZE;

  threadgroup T scales_block[BM * groups_per_block];
  threadgroup T biases_block[BM * groups_per_block];
  threadgroup T x_block[colgroup];

  thread uint32_t w_local[TM];
  thread T result[TM] = {0};
  thread T scale = 1;
  thread T bias = 0;
  thread T x_thread[el_per_thread];

  // Adjust positions
  int out_row = tid.y * BM * TM + simd_gid * TM;
  w += out_row * in_vec_size / el_per_thread;
  scales += out_row * (in_vec_size / groups);
  biases += out_row * (in_vec_size / groups);
  x += tid.z * in_vec_size;
  y += tid.z * out_vec_size;

  // Loop over in_vec in blocks of colgroup
  for (int i=0; i<in_vec_size; i+=colgroup) {
    // Load the vec to shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid < simdgroups_fetching_vec) {
      x_block[lid] = x[lid + i];
    }
    if (simd_lid == 0) {
      #pragma clang loop unroll(full)
      for (int j=0; j<groups_per_block; j++) {
        scales_block[simd_gid * groups_per_block + j] = scales[i / groups + j];
      }
      #pragma clang loop unroll(full)
      for (int j=0; j<groups_per_block; j++) {
        biases_block[simd_gid * groups_per_block + j] = biases[i / groups + j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load in_vec, scale, bias to registers
    #pragma clang loop unroll(full)
    for (int j=0; j<el_per_thread; j++) {
      x_thread[j] = x_block[simd_lid*el_per_thread + j];
    }
    scale = scales_block[simd_lid * el_per_thread / groups];
    bias = biases_block[simd_lid * el_per_thread / groups];

    // Load the matrix elements
    #pragma clang loop unroll(full)
    for (int j=0; j<TM; j++) {
      w_local[j] = w[i / el_per_thread + simd_lid + j * in_vec_size / el_per_thread];
    }

    // Do all the work.
    #pragma clang loop unroll(full)
    for (int j=0; j<TM; j++) {
      #pragma clang loop unroll(full)
      for (int k=0; k<el_per_thread; k++) {
        result[j] += (scale * static_cast<T>(w_local[j] & bitmask) + bias) * x_thread[k];
        w_local[j] >>= width;
      }
    }
  }

  // Accumulate in the simdgroup
  for (int j=0; j<TM; j++) {
    result[j] = simd_sum(result[j]);
  }

  // Store the result
  if (simd_lid == 0) {
    for (int j=0; j<TM; j++) {
      y[out_row + j] = result[j];
    }
  }
}


#define instantiate_qmv(name, itype, groups, width) \
  template [[host_name("qmv_n_" #name "_groups_" #groups "_width_" #width)]] \
  [[kernel]] void qmv<itype, 32, 32, groups, width>( \
    const device uint32_t* w [[buffer(0)]], \
    const device itype* scales [[buffer(0)]], \
    const device itype* biases [[buffer(1)]], \
    const device itype* x [[buffer(2)]], \
    device itype* y [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_qmv(float32, float, 128, 2);
instantiate_qmv(float16, half, 128, 2);
instantiate_qmv(bfloat16, bfloat, 128, 2);
instantiate_qmv(float32, float, 64, 2);
instantiate_qmv(float16, half, 64, 2);
instantiate_qmv(bfloat16, bfloat, 64, 2);
instantiate_qmv(float32, float, 128, 4);
instantiate_qmv(float16, half, 128, 4);
instantiate_qmv(bfloat16, bfloat, 128, 4);
instantiate_qmv(float32, float, 64, 4);
instantiate_qmv(float16, half, 64, 4);
instantiate_qmv(bfloat16, bfloat, 64, 4);
instantiate_qmv(float32, float, 128, 8);
instantiate_qmv(float16, half, 128, 8);
instantiate_qmv(bfloat16, bfloat, 128, 8);
instantiate_qmv(float32, float, 64, 8);
instantiate_qmv(float16, half, 64, 8);
instantiate_qmv(bfloat16, bfloat, 64, 8);
