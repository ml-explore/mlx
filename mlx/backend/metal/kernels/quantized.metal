// Copyright © 2023 Apple Inc.

#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/gemm/gemm.h"
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


template <typename T, const int BM, const int BK, const int BN, const int groups, const int width>
[[kernel]] void qmm_t(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  const uint lidy = lid / SIMD_SIZE;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int bitmask = (1 << width) - 1;
  constexpr int el_per_int = 32 / width;
  constexpr int ints_per_block = BK / el_per_int;
  constexpr int groups_per_block = (BK / groups > 0) ? (BK / groups) : 1;
  constexpr int groups_per_simd = BN / (WM * WN);
  constexpr int w_els_per_thread = (BN * BK / el_per_int) / (SIMD_SIZE * WM * WN);

  // Using the kernel just as a type to instantiate the appropriate BlockMMA
  // and constexpr size calculations
  using mma_t = BlockMMA<T, BM, BN, BK, WM, WN, false, true>;
  using loader_x_t = BlockLoader<T, BM, BK, BK, 4, WM * WN * SIMD_SIZE, false, true, 0>;

  threadgroup T scales_block[BN * groups_per_block];
  threadgroup T biases_block[BN * groups_per_block];
  threadgroup T Xs[BN * BK];
  threadgroup T Ws[BM * BK];

  // Set the block
  const int K_w = K / el_per_int;
  const int K_g = K / groups;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * K;
  w += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * N + y_col;

  // Make the x loader and mma operation
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k=0; k<K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Load the x tile
    loader_x.load_unsafe();

    // Load the scale and bias
    if (simd_lid == 0) {
      threadgroup T *scales_block_local = scales_block + lidy * groups_per_block * groups_per_simd;
      threadgroup T *biases_block_local = biases_block + lidy * groups_per_block * groups_per_simd;
      const device T *scales_local = scales + lidy * groups_per_simd * K_g + k / groups;
      const device T *biases_local = biases + lidy * groups_per_simd * K_g + k / groups;
      #pragma clang loop unroll(full)
      for (int gs=0; gs<groups_per_simd; gs++) {
        #pragma clang loop unroll(full)
        for (int gc=0; gc<groups_per_block; gc++) {
          scales_block_local[gc] = scales_local[gc];
          biases_block_local[gc] = biases_local[gc];
        }
        scales_block_local += groups_per_block;
        scales_local += K_g;
        biases_block_local += groups_per_block;
        biases_local += K_g;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load the w tile
    {
      for (int wo=0; wo<w_els_per_thread; wo++) {
        int offset = lid * w_els_per_thread + wo;
        int offset_row = offset / (BK / el_per_int);
        int offset_col = offset % (BK / el_per_int);
        const device uint32_t * w_local = w + offset_row * K_w + offset_col;
        threadgroup T * Ws_local = Ws + offset_row * BK + offset_col * el_per_int;

        uint32_t wi = *w_local;
        T scale = scales_block[offset_row * groups_per_block + offset_col / (groups / el_per_int)];
        T bias = biases_block[offset_row * groups_per_block + offset_col / (groups / el_per_int)];

        #pragma clang loop unroll(full)
        for (int t=0; t<el_per_int; t++) {
          Ws_local[t] = scale * static_cast<T>(wi & bitmask) + bias;
          wi >>= width;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(Xs, Ws);

    // Prepare for next iteration
    loader_x.next();
    w += ints_per_block;
    // scales and biases cannot be advanced because they would have to be
    // advanced every other iteration or sth.
  }

  // Store results to device memory
  mma_op.store_result(y, N);
}


#define instantiate_qmv(name, itype, groups, width) \
  template [[host_name("qmv_n_" #name "_groups_" #groups "_width_" #width)]] \
  [[kernel]] void qmv<itype, 32, 32, groups, width>( \
    const device uint32_t* w [[buffer(0)]], \
    const device itype* scales [[buffer(1)]], \
    const device itype* biases [[buffer(2)]], \
    const device itype* x [[buffer(3)]], \
    device itype* y [[buffer(4)]], \
    const constant int& in_vec_size [[buffer(5)]], \
    const constant int& out_vec_size [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_qmv_types(groups, width) \
  instantiate_qmv(float32, float, groups, width) \
  instantiate_qmv(float16, half, groups, width) \
  instantiate_qmv(bfloat16, bfloat16_t, groups, width)

instantiate_qmv_types(128, 2)
instantiate_qmv_types(128, 4)
instantiate_qmv_types(128, 8)
instantiate_qmv_types( 64, 2)
instantiate_qmv_types( 64, 4)
instantiate_qmv_types( 64, 8)

#define instantiate_qmm_t(name, itype, groups, width) \
  template [[host_name("qmm_t_" #name "_groups_" #groups "_width_" #width)]] \
  [[kernel]] void qmm_t<itype, 32, 64, 32, groups, width>( \
      const device itype* x [[buffer(0)]], \
      const device uint32_t* w [[buffer(1)]], \
      const device itype* scales [[buffer(2)]], \
      const device itype* biases [[buffer(3)]], \
      device itype* y [[buffer(4)]], \
      const constant int& M [[buffer(5)]], \
      const constant int& N [[buffer(6)]], \
      const constant int& K [[buffer(7)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint lid [[thread_index_in_threadgroup]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_qmm_t_types(groups, width) \
  instantiate_qmm_t(float32, float, groups, width) \
  instantiate_qmm_t(float16, half, groups, width) \
  instantiate_qmm_t(bfloat16, bfloat16_t, groups, width)

instantiate_qmm_t_types(128, 2)
instantiate_qmm_t_types(128, 4)
instantiate_qmm_t_types(128, 8)
instantiate_qmm_t_types( 64, 2)
instantiate_qmm_t_types( 64, 4)
instantiate_qmm_t_types( 64, 8)
