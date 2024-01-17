// Copyright © 2023 Apple Inc.

#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

template <typename T, const int BM, const int BN, const int group_size, const int bits>
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

  constexpr int bitmask = (1 << bits) - 1;
  constexpr int el_per_thread = 32 / bits;
  constexpr int colgroup = BN * el_per_thread;
  constexpr int groups_per_block = colgroup / group_size;
  constexpr int simdgroups_fetching_vec = colgroup / SIMD_SIZE;

  threadgroup T scales_block[BM * groups_per_block];
  threadgroup T biases_block[BM * groups_per_block];
  threadgroup T x_block[colgroup];

  thread uint32_t w_local;
  thread T result = 0;
  thread T scale = 1;
  thread T bias = 0;
  thread T x_thread[el_per_thread];

  // Adjust positions
  const int in_vec_size_w = in_vec_size / el_per_thread;
  const int in_vec_size_g = in_vec_size / group_size;
  int out_row = tid.y * BM + simd_gid;
  w += out_row * in_vec_size_w;
  scales += out_row * in_vec_size_g;
  biases += out_row * in_vec_size_g;
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
        scales_block[simd_gid * groups_per_block + j] = scales[i / group_size + j];
      }
      #pragma clang loop unroll(full)
      for (int j=0; j<groups_per_block; j++) {
        biases_block[simd_gid * groups_per_block + j] = biases[i / group_size + j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load in_vec, scale, bias to registers
    #pragma clang loop unroll(full)
    for (int j=0; j<el_per_thread; j++) {
      x_thread[j] = x_block[simd_lid*el_per_thread + j];
    }
    scale = scales_block[simd_gid * groups_per_block + simd_lid * el_per_thread / group_size];
    bias = biases_block[simd_gid * groups_per_block + simd_lid * el_per_thread / group_size];

    // Load the matrix elements
    w_local = w[i / el_per_thread + simd_lid];

    // Do all the work.
    #pragma clang loop unroll(full)
    for (int k=0; k<el_per_thread; k++) {
      result += (scale * static_cast<T>(w_local & bitmask) + bias) * x_thread[k];
      w_local >>= bits;
    }
  }

  // Accumulate in the simdgroup
  result = simd_sum(result);

  // Store the result
  if (simd_lid == 0) {
    y[out_row] = result;
  }
}


template <typename T, const int BM, const int BN, const int group_size, const int bits>
[[kernel]] void qvm(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  static_assert(BM == SIMD_SIZE, "qvm expects BM to be equal to SIMD_SIZE");
  static_assert(BN == BM, "qvm expects a block size of 32x32");

  (void)lid;

  constexpr int bitmask = (1 << bits) - 1;
  constexpr int el_per_int = 32 / bits;
  constexpr int colgroup = BN * el_per_int;
  constexpr int groups_per_block = colgroup / group_size;

  threadgroup T scales_block[BM * groups_per_block];
  threadgroup T biases_block[BM * groups_per_block];
  threadgroup T x_block[BM];

  thread uint32_t w_local;
  thread T result[el_per_int] = {0};
  thread T scale = 1;
  thread T bias = 0;
  thread T x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size / el_per_int;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = (tid.y * BN + simd_gid) * el_per_int;
  w += out_col / el_per_int;
  scales += out_col / group_size;
  biases += out_col / group_size;
  x += tid.z * in_vec_size;
  y += tid.z * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of colgroup
  for (int i=0; i<in_vec_size; i+=BM) {
    int offset = simd_lid + i;
    bool thread_in_bounds = offset < in_vec_size;

    // Load the vec to shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
      x_block[simd_lid] = (thread_in_bounds) ? x[offset] : 0;
    }

    // Load the scales and biases to shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
      #pragma clang loop unroll(full)
      for (int j=0; j<groups_per_block; j++) {
        scales_block[simd_lid * groups_per_block + j] = scales[(i + simd_lid) * out_vec_size_g + j];
      }
      #pragma clang loop unroll(full)
      for (int j=0; j<groups_per_block; j++) {
        biases_block[simd_lid * groups_per_block + j] = biases[(i + simd_lid) * out_vec_size_g + j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load in_vec, scale, bias to registers
    x_local = x_block[simd_lid];
    scale = scales_block[simd_lid * groups_per_block + (simd_gid * el_per_int) / group_size];
    bias = biases_block[simd_lid * groups_per_block + (simd_gid * el_per_int) / group_size];

    // Load the matrix elements
    w_local = (thread_in_bounds) ? w[offset * out_vec_size_w] : 0;

    // Do all the work.
    #pragma clang loop unroll(full)
    for (int k=0; k<el_per_int; k++) {
      result[k] += (scale * static_cast<T>(w_local & bitmask) + bias) * x_local;
      w_local >>= bits;
    }
  }

  // Accumulate in the simdgroup
  #pragma clang loop unroll(full)
  for (int k=0; k<el_per_int; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
    #pragma clang loop unroll(full)
    for (int k=0; k<el_per_int; k++) {
      y[k] = result[k];
    }
  }
}


template <typename T, const int BM, const int BK, const int BN, const int group_size, const int bits, const bool aligned_N>
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
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int el_per_int = 32 / bits;
  constexpr int ints_per_block = BK / el_per_int;
  constexpr int groups_per_block = (BK / group_size > 0) ? (BK / group_size) : 1;
  constexpr int groups_per_simd = BN / (WM * WN);
  constexpr int w_els_per_thread = (BN * BK / el_per_int) / (SIMD_SIZE * WM * WN);

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK, BK>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK, 1, WM * WN * SIMD_SIZE, 1, 4>;


  threadgroup T scales_block[BN * groups_per_block];
  threadgroup T biases_block[BN * groups_per_block];
  threadgroup T Xs[BM * BK];
  threadgroup T Ws[BN * BK];

  // Set the block
  const int K_w = K / el_per_int;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * K;
  w += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * N + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k=0; k<K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Load the x tile
    if (num_els < BM) {
        loader_x.load_safe(short2(BK, num_els));
    } else {
        loader_x.load_unsafe();
    }

    // Load the scale and bias
    if (simd_lid == 0) {
      threadgroup T *scales_block_local = scales_block + lidy * groups_per_block * groups_per_simd;
      threadgroup T *biases_block_local = biases_block + lidy * groups_per_block * groups_per_simd;
      const device T *scales_local = scales + lidy * groups_per_simd * K_g + k / group_size;
      const device T *biases_local = biases + lidy * groups_per_simd * K_g + k / group_size;
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
      if (!aligned_N && num_outs < BN) {
        for (int wo=0; wo<w_els_per_thread; wo++) {
          int offset = lid * w_els_per_thread + wo;
          int offset_row = offset / (BK / el_per_int);
          int offset_col = offset % (BK / el_per_int);
          const device uint32_t * w_local = w + offset_row * K_w + offset_col;
          threadgroup T * Ws_local = Ws + offset_row * BK + offset_col * el_per_int;

          if (y_col + offset_col < N) {
            uint32_t wi = *w_local;
            T scale = scales_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];
            T bias = biases_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];

            #pragma clang loop unroll(full)
            for (int t=0; t<el_per_int; t++) {
              Ws_local[t] = scale * static_cast<T>(wi & bitmask) + bias;
              wi >>= bits;
            }
          } else {
            #pragma clang loop unroll(full)
            for (int t=0; t<el_per_int; t++) {
              Ws_local[t] = 0;
            }
          }
        }
      } else {
        for (int wo=0; wo<w_els_per_thread; wo++) {
          int offset = lid * w_els_per_thread + wo;
          int offset_row = offset / (BK / el_per_int);
          int offset_col = offset % (BK / el_per_int);
          const device uint32_t * w_local = w + offset_row * K_w + offset_col;
          threadgroup T * Ws_local = Ws + offset_row * BK + offset_col * el_per_int;

          uint32_t wi = *w_local;
          T scale = scales_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];
          T bias = biases_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];

          #pragma clang loop unroll(full)
          for (int t=0; t<el_per_int; t++) {
            Ws_local[t] = scale * static_cast<T>(wi & bitmask) + bias;
            wi >>= bits;
          }
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}


template <typename T, const int BM, const int BK, const int BN, const int group_size, const int bits>
[[kernel]] void qmm_n(
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
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int el_per_int = 32 / bits;
  constexpr int groups_per_block = (BN / group_size > 0) ? (BN / group_size) : 1;
  constexpr int groups_per_simd = BK / (WM * WN);
  constexpr int w_els_per_thread = (BK * BN / el_per_int) / (SIMD_SIZE * WM * WN);

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK, BN>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK, 1, WM * WN * SIMD_SIZE, 1, 4>;

  threadgroup T scales_block[BK * groups_per_block];
  threadgroup T biases_block[BK * groups_per_block];
  threadgroup T Xs[BM * BK];
  threadgroup T Ws[BK * BN];

  // Set the block
  const int N_w = N / el_per_int;
  const int N_g = N / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * K;
  w += y_col / el_per_int;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * N + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k=0; k<K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Load the x tile
    if (num_els < BM) {
        loader_x.load_safe(short2(BK, num_els));
    } else {
        loader_x.load_unsafe();
    }

    // Load the scale and bias
    if (simd_lid == 0) {
      threadgroup T *scales_block_local = scales_block + lidy * groups_per_block * groups_per_simd;
      threadgroup T *biases_block_local = biases_block + lidy * groups_per_block * groups_per_simd;
      const device T *scales_local = scales + lidy * groups_per_simd * N_g;
      const device T *biases_local = biases + lidy * groups_per_simd * N_g;
      #pragma clang loop unroll(full)
      for (int gs=0; gs<groups_per_simd; gs++) {
        #pragma clang loop unroll(full)
        for (int gc=0; gc<groups_per_block; gc++) {
          scales_block_local[gc] = scales_local[gc];
          biases_block_local[gc] = biases_local[gc];
        }
        scales_block_local += groups_per_block;
        scales_local += N_g;
        biases_block_local += groups_per_block;
        biases_local += N_g;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load the w tile
    {
      if (k + BK >= K) {
        for (int wo=0; wo<w_els_per_thread; wo++) {
          int offset = lid * w_els_per_thread + wo;
          int offset_row = offset / (BN / el_per_int);
          int offset_col = offset % (BN / el_per_int);
          const device uint32_t * w_local = w + offset_row * N_w + offset_col;
          threadgroup T * Ws_local = Ws + offset_row * BN + offset_col * el_per_int;

          if (y_row + offset_row < K) {
            uint32_t wi = *w_local;
            T scale = scales_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];
            T bias = biases_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];

            #pragma clang loop unroll(full)
            for (int t=0; t<el_per_int; t++) {
              Ws_local[t] = scale * static_cast<T>(wi & bitmask) + bias;
              wi >>= bits;
            }
          } else {
            #pragma clang loop unroll(full)
            for (int t=0; t<el_per_int; t++) {
              Ws_local[t] = 0;
            }
          }
        }
      } else {
        for (int wo=0; wo<w_els_per_thread; wo++) {
          int offset = lid * w_els_per_thread + wo;
          int offset_row = offset / (BN / el_per_int);
          int offset_col = offset % (BN / el_per_int);
          const device uint32_t * w_local = w + offset_row * N_w + offset_col;
          threadgroup T * Ws_local = Ws + offset_row * BN + offset_col * el_per_int;

          uint32_t wi = *w_local;
          T scale = scales_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];
          T bias = biases_block[offset_row * groups_per_block + offset_col / (group_size / el_per_int)];

          #pragma clang loop unroll(full)
          for (int t=0; t<el_per_int; t++) {
            Ws_local[t] = scale * static_cast<T>(wi & bitmask) + bias;
            wi >>= bits;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(Xs, Ws);

    // Prepare for next iteration
    loader_x.next();
    w += BK * N_w;
    scales += BK * N_g;
    biases += BK * N_g;
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}


#define instantiate_qmv(name, itype, group_size, bits) \
  template [[host_name("qmv_" #name "_gs_" #group_size "_b_" #bits)]] \
  [[kernel]] void qmv<itype, 32, 32, group_size, bits>( \
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

#define instantiate_qmv_types(group_size, bits) \
  instantiate_qmv(float32, float, group_size, bits) \
  instantiate_qmv(float16, half, group_size, bits) \
  instantiate_qmv(bfloat16, bfloat16_t, group_size, bits)

instantiate_qmv_types(128, 2)
instantiate_qmv_types(128, 4)
instantiate_qmv_types(128, 8)
instantiate_qmv_types( 64, 2)
instantiate_qmv_types( 64, 4)
instantiate_qmv_types( 64, 8)

#define instantiate_qvm(name, itype, group_size, bits) \
  template [[host_name("qvm_" #name "_gs_" #group_size "_b_" #bits)]] \
  [[kernel]] void qvm<itype, 32, 32, group_size, bits>( \
    const device itype* x [[buffer(0)]], \
    const device uint32_t* w [[buffer(1)]], \
    const device itype* scales [[buffer(2)]], \
    const device itype* biases [[buffer(3)]], \
    device itype* y [[buffer(4)]], \
    const constant int& in_vec_size [[buffer(5)]], \
    const constant int& out_vec_size [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_qvm_types(group_size, bits) \
  instantiate_qvm(float32, float, group_size, bits) \
  instantiate_qvm(float16, half, group_size, bits) \
  instantiate_qvm(bfloat16, bfloat16_t, group_size, bits)

instantiate_qvm_types(128, 2)
instantiate_qvm_types(128, 4)
instantiate_qvm_types(128, 8)
instantiate_qvm_types( 64, 2)
instantiate_qvm_types( 64, 4)
instantiate_qvm_types( 64, 8)

#define instantiate_qmm_t(name, itype, group_size, bits, aligned_N) \
  template [[host_name("qmm_t_" #name "_gs_" #group_size "_b_" #bits "_alN_" #aligned_N)]] \
  [[kernel]] void qmm_t<itype, 32, 64, 32, group_size, bits, aligned_N>( \
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

#define instantiate_qmm_t_types(group_size, bits) \
  instantiate_qmm_t(float32, float, group_size, bits, false) \
  instantiate_qmm_t(float16, half, group_size, bits, false) \
  instantiate_qmm_t(bfloat16, bfloat16_t, group_size, bits, false) \
  instantiate_qmm_t(float32, float, group_size, bits, true) \
  instantiate_qmm_t(float16, half, group_size, bits, true) \
  instantiate_qmm_t(bfloat16, bfloat16_t, group_size, bits, true)

instantiate_qmm_t_types(128, 2)
instantiate_qmm_t_types(128, 4)
instantiate_qmm_t_types(128, 8)
instantiate_qmm_t_types( 64, 2)
instantiate_qmm_t_types( 64, 4)
instantiate_qmm_t_types( 64, 8)

#define instantiate_qmm_n(name, itype, group_size, bits) \
  template [[host_name("qmm_n_" #name "_gs_" #group_size "_b_" #bits)]] \
  [[kernel]] void qmm_n<itype, 32, 32, 64, group_size, bits>( \
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

#define instantiate_qmm_n_types(group_size, bits) \
  instantiate_qmm_n(float32, float, group_size, bits) \
  instantiate_qmm_n(float16, half, group_size, bits) \
  instantiate_qmm_n(bfloat16, bfloat16_t, group_size, bits)

instantiate_qmm_n_types(128, 2)
instantiate_qmm_n_types(128, 4)
instantiate_qmm_n_types(128, 8)
instantiate_qmm_n_types( 64, 2)
instantiate_qmm_n_types( 64, 4)
instantiate_qmm_n_types( 64, 8)
