// Copyright © 2023-2024 Apple Inc.

#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

template <typename T> struct AccT {
  typedef T acc_t;
};

template <> struct AccT<bfloat16_t> {
  typedef float acc_t;
};


template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T *x, thread U *x_thread) {
  static_assert(bits == 2 || bits == 4 || bits == 8, "Template undefined for bits not in {2, 4, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i+1] + x[i+2] + x[i+3];
      x_thread[i] = x[i];
      x_thread[i+1] = x[i+1] / 4.0f;
      x_thread[i+2] = x[i+2] / 16.0f;
      x_thread[i+3] = x[i+3] / 64.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i+1] + x[i+2] + x[i+3];
      x_thread[i] = x[i];
      x_thread[i+1] = x[i+1] / 16.0f;
      x_thread[i+2] = x[i+2] / 256.0f;
      x_thread[i+3] = x[i+3] / 4096.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(const device uint8_t* w, const thread U *x_thread, U scale, U bias, U sum) {
  static_assert(bits == 2 || bits == 4 || bits == 8, "Template undefined for bits not in {2, 4, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (
          x_thread[4*i] * (w[i] & 0x03)
          + x_thread[4*i+1] * (w[i] & 0x0c)
          + x_thread[4*i+2] * (w[i] & 0x30)
          + x_thread[4*i+3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (
          x_thread[4*i] * (ws[i] & 0x000f)
          + x_thread[4*i+1] * (ws[i] & 0x00f0)
          + x_thread[4*i+2] * (ws[i] & 0x0f00)
          + x_thread[4*i+3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(bits == 2 || bits == 4 || bits == 8, "Template undefined for bits not in {2, 4, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4*i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4*i+1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4*i+2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4*i+3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
    U s[4] = {scale, scale / 16.0f, scale / 256.0f, scale / 4096.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4*i] += x * (s[0] * (ws[i] & 0x000f) + bias);
      result[4*i+1] += x * (s[1] * (ws[i] & 0x00f0) + bias);
      result[4*i+2] += x * (s[2] * (ws[i] & 0x0f00) + bias);
      result[4*i+3] += x * (s[3] * (ws[i] & 0xf000) + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <typename T, int group_size, int bits, int packs_per_thread>
[[kernel]] void qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  w += out_row * in_vec_size_w + simd_lid * packs_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.z * in_vec_size + simd_lid * values_per_thread;
  y += tid.z * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = (const device uint8_t *)(w + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    w += block_size / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}


template <typename T, const int group_size, const int bits>
[[kernel]] void qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    w += out_row * in_vec_size_w + simd_lid * packs_per_thread;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.z * in_vec_size + simd_lid * values_per_thread;
    y += tid.z * out_vec_size + out_row;

    for (int k = 0; k < in_vec_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; out_row + row < out_vec_size; row++) {
        const device uint8_t* wl = (const device uint8_t *)(w + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      w += block_size / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }

    for (int row = 0; out_row + row < out_vec_size; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    w += used_out_row * in_vec_size_w + simd_lid * packs_per_thread;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.z * in_vec_size + simd_lid * values_per_thread;
    y += tid.z * out_vec_size + used_out_row;

    for (int k = 0; k < in_vec_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* wl = (const device uint8_t *)(w + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      w += block_size / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}


template <typename T, const int group_size, const int bits>
[[kernel]] void qvm(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int num_simdgroups = 8;
  constexpr int pack_factor = 32 / bits;
  constexpr int blocksize = SIMD_SIZE;

  typedef float U;

  thread uint32_t w_local;
  thread U result[pack_factor] = {0};
  thread U scale = 1;
  thread U bias = 0;
  thread U x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = tid.y * (num_simdgroups * pack_factor) + simd_gid * pack_factor;
  w += out_col / pack_factor;
  scales += out_col / group_size;
  biases += out_col / group_size;
  x += tid.z * in_vec_size;
  y += tid.z * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of blocksize
  int i = 0;
  for (; i + blocksize <= in_vec_size; i += blocksize) {
    x_local = x[i + simd_lid];
    scale = scales[(i + simd_lid) * out_vec_size_g];
    bias = biases[(i + simd_lid) * out_vec_size_g];
    w_local = w[(i + simd_lid) * out_vec_size_w];

    qouter<U, pack_factor, bits>((thread uint8_t *)&w_local, x_local, scale, bias, result);
  }
  if (static_cast<int>(i + simd_lid) < in_vec_size) {
    x_local = x[i + simd_lid];
    scale = scales[(i + simd_lid) * out_vec_size_g];
    bias = biases[(i + simd_lid) * out_vec_size_g];
    w_local = w[(i + simd_lid) * out_vec_size_w];
  } else {
    x_local = 0;
    scale = 0;
    bias = 0;
    w_local = 0;
  }
  qouter<U, pack_factor, bits>((thread uint8_t *)&w_local, x_local, scale, bias, result);

  // Accumulate in the simdgroup
  #pragma clang loop unroll(full)
  for (int k=0; k<pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
    #pragma clang loop unroll(full)
    for (int k=0; k<pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
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

          if (y_row + offset_row < N) {
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
    short num_k = min(BK, K - k);
    if (num_els < BM || num_k < BK) {
        loader_x.load_safe(short2(num_k, num_els));
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
      if (num_k < BK) {
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


#define instantiate_qmv_fast(name, itype, group_size, bits, packs_per_thread) \
  template [[host_name("qmv_" #name "_gs_" #group_size "_b_" #bits "_fast")]] \
  [[kernel]] void qmv_fast<itype, group_size, bits, packs_per_thread>( \
    const device uint32_t* w [[buffer(0)]], \
    const device itype* scales [[buffer(1)]], \
    const device itype* biases [[buffer(2)]], \
    const device itype* x [[buffer(3)]], \
    device itype* y [[buffer(4)]], \
    const constant int& in_vec_size [[buffer(5)]], \
    const constant int& out_vec_size [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_qmv_fast_types(group_size, bits, packs_per_thread) \
  instantiate_qmv_fast(float32, float, group_size, bits, packs_per_thread) \
  instantiate_qmv_fast(float16, half, group_size, bits, packs_per_thread) \
  instantiate_qmv_fast(bfloat16, bfloat16_t, group_size, bits, packs_per_thread)

instantiate_qmv_fast_types(128, 2, 1)
instantiate_qmv_fast_types(128, 4, 2)
instantiate_qmv_fast_types(128, 8, 2)
instantiate_qmv_fast_types( 64, 2, 1)
instantiate_qmv_fast_types( 64, 4, 2)
instantiate_qmv_fast_types( 64, 8, 2)
instantiate_qmv_fast_types( 32, 2, 1)
instantiate_qmv_fast_types( 32, 4, 2)
instantiate_qmv_fast_types( 32, 8, 2)

#define instantiate_qmv(name, itype, group_size, bits) \
  template [[host_name("qmv_" #name "_gs_" #group_size "_b_" #bits)]] \
  [[kernel]] void qmv<itype, group_size, bits>( \
    const device uint32_t* w [[buffer(0)]], \
    const device itype* scales [[buffer(1)]], \
    const device itype* biases [[buffer(2)]], \
    const device itype* x [[buffer(3)]], \
    device itype* y [[buffer(4)]], \
    const constant int& in_vec_size [[buffer(5)]], \
    const constant int& out_vec_size [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
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
instantiate_qmv_types( 32, 2)
instantiate_qmv_types( 32, 4)
instantiate_qmv_types( 32, 8)

#define instantiate_qvm(name, itype, group_size, bits) \
  template [[host_name("qvm_" #name "_gs_" #group_size "_b_" #bits)]] \
  [[kernel]] void qvm<itype, group_size, bits>( \
    const device itype* x [[buffer(0)]], \
    const device uint32_t* w [[buffer(1)]], \
    const device itype* scales [[buffer(2)]], \
    const device itype* biases [[buffer(3)]], \
    device itype* y [[buffer(4)]], \
    const constant int& in_vec_size [[buffer(5)]], \
    const constant int& out_vec_size [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
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
instantiate_qvm_types( 32, 2)
instantiate_qvm_types( 32, 4)
instantiate_qvm_types( 32, 8)

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
instantiate_qmm_t_types( 32, 2)
instantiate_qmm_t_types( 32, 4)
instantiate_qmm_t_types( 32, 8)

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
instantiate_qmm_n_types( 32, 2)
instantiate_qmm_n_types( 32, 4)
instantiate_qmm_n_types( 32, 8)
