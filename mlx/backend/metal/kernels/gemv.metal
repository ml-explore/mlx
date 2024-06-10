// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

struct _NoMask {
  char x;

  constexpr METAL_FUNC operator bool() {
    return true;
  }
  constexpr METAL_FUNC operator bool() const threadgroup {
    return true;
  }
  constexpr METAL_FUNC operator bool() const device {
    return true;
  }
  constexpr METAL_FUNC operator bool() const constant {
    return true;
  }
};

typedef struct _NoMask nomask_t;

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    typename out_mask_t = nomask_t, /* Do out masking */
    typename op_mask_t = nomask_t> /* Do op masking */
struct GEMVKernel {
  static_assert(
      BN == 8 || BN == 16 || BN == 32,
      "gemv block must have a width of 8, 16, or 32");

  // - The matrix of size (M = out_vec_size, N = in_vec_size) is divided up
  //   into blocks of (BM * TM, BN * TN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each thead group is launched with (BN, BM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows
  //    and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for
  //    the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN * TN * 2;

  MLX_MTL_CONST bool has_operand_mask = !metal::is_same_v<op_mask_t, nomask_t>;
  MLX_MTL_CONST bool has_output_mask = !metal::is_same_v<out_mask_t, nomask_t>;

  MLX_MTL_CONST bool has_mul_operand_mask =
      has_operand_mask && !metal::is_same_v<op_mask_t, bool>;
  MLX_MTL_CONST bool has_mul_output_mask =
      has_output_mask && !metal::is_same_v<out_mask_t, bool>;

  static METAL_FUNC void
  load_unsafe(const device T* src, thread T dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = src[src_offset + tn];
    }
  }

  static METAL_FUNC void load_safe(
      const device T* src,
      thread T dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = src[src_offset + tn];
    }

    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src[src_offset + tn];
      }
    } else { // Edgecase
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size ? src[src_offset + tn] : 0;
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      const device T* bias [[buffer(2)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& matrix_ld [[buffer(6)]],
      const constant float& alpha [[buffer(7)]],
      const constant float& beta [[buffer(8)]],
      const constant int& bias_stride [[buffer(14)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    thread T result[TM] = {0};
    thread T inter[TN];
    thread T v_coeff[TN];

    // Block position
    int out_row = (tid.x * BM + simd_gid) * TM;

    // Exit simdgroup if rows out of bound
    if (out_row >= out_vec_size)
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Advance matrix
    mat += out_row * matrix_ld;

    constexpr const uniform<int> loop_stride = make_uniform(BN * TN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    int bn = simd_lid * TN;

    // Loop over in_vec in blocks of BN * TN
    for (int i = 0; i < n_iter; ++i) {
      load_unsafe(in_vec, v_coeff, bn);

      // Per thread work loop
      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_unsafe(mat, inter, mat_offset + bn);

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }

        mat_offset += matrix_ld;
      }

      bn += BN * TN;
    }

    if (leftover > 0) {
      if (bn + TN <= in_size) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          v_coeff[tn] = in_vec[bn + tn];
        }
      } else { // Edgecase
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          v_coeff[tn] = bn + tn < in_size ? in_vec[bn + tn] : 0;
        }
      }

      // Per thread work loop
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        if (bn + TN <= in_size) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[tm * matrix_ld + bn + tn];
          }

        } else { // Edgecase
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            int col_idx = (bn + tn) < in_size ? (bn + tn) : (leftover - 1);
            inter[tn] = mat[tm * matrix_ld + col_idx];
          }
        }

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort si = (BN / 2); si >= 1; si >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], si);
      }
    }

    // Write outputs
    if (simd_lid == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        if (kDoAxpby) {
          out_vec[out_row + tm] = static_cast<T>(alpha) * result[tm] +
              static_cast<T>(beta) * bias[(out_row + tm) * bias_stride];
        } else {
          out_vec[out_row + tm] = result[tm];
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
struct GEMVTKernel {
  // - The matrix of size (M = in_vec_size, N = out_vec_size) is divided up
  //   into blocks of (BM * TM, BN * TN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each thead group is launched with (BN, BM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows
  //    and the corresponding scalar from the vector
  // 2. The thread then accumulates its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN * BM * TN;

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      const device T* bias [[buffer(2)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& marix_ld [[buffer(6)]],
      const constant float& alpha [[buffer(7)]],
      const constant float& beta [[buffer(8)]],
      const constant int& bias_stride [[buffer(14)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)simd_gid;
    (void)simd_lid;

    // Thread local accumulation results
    T result[TN] = {0};
    T inter[TN];
    T v_coeff[TM];

    // Threadgroup accumulation results
    threadgroup T* tgp_results = tgp_memory + lid.x * BM * TN;

    int out_col = (tid.x * BN + lid.x) * TN;
    int in_row = lid.y * TM;

    // Edgecase handling
    if (out_col < out_vec_size) {
      out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;

      // Per thread accumulation main loop
      int bm = in_row;
      for (; bm < in_vec_size; bm += BM * TM) {
        // Adding a threadgroup_barrier improves performance slightly
        // This is possibly it may help exploit cache better
        threadgroup_barrier(mem_flags::mem_none);

        if (bm + TM <= in_vec_size) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            v_coeff[tm] = in_vec[bm + tm];
          }

          MLX_MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            for (int tn = 0; tn < TN; tn++) {
              inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
            }
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += v_coeff[tm] * inter[tn];
            }
          }

        } else { // Edgecase handling
          for (int tm = 0; bm + tm < in_vec_size; tm++) {
            v_coeff[tm] = in_vec[bm + tm];

            for (int tn = 0; tn < TN; tn++) {
              inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
            }
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += v_coeff[tm] * inter[tn];
            }
          }
        }
      }
    }

    // Threadgroup collection

    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < TN; i++) {
      tgp_results[lid.y * TN + i] = result[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup accumulation and writing out results
    if (lid.y == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 1; i < BM; i++) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          result[j] += tgp_results[i * TN + j];
        }
      }

      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        if (kDoAxpby) {
          out_vec[out_col + j] = static_cast<T>(alpha) * result[j] +
              static_cast<T>(beta) * bias[(out_col + j) * bias_stride];
        } else {
          out_vec[out_col + j] = result[j];
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
[[kernel, max_total_threads_per_threadgroup(BM* BN)]] void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant size_t* vector_batch_stride [[buffer(11)]],
    const constant size_t* matrix_batch_stride [[buffer(12)]],
    const constant size_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVKernel<T, BM, BN, TM, TN, kDoAxpby>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (kDoAxpby) {
      bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (kDoAxpby) {
      bias += tid.z * bias_batch_stride[0];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      bias_stride,
      tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

#define instantiate_gemv_helper(name, itype, bm, bn, tm, tn, nc, axpby)      \
  template [[host_name("gemv_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn \
                       "_nc" #nc "_axpby" #axpby)]] [[kernel]] void          \
  gemv<itype, bm, bn, tm, tn, nc, axpby>(                                    \
      const device itype* mat [[buffer(0)]],                                 \
      const device itype* in_vec [[buffer(1)]],                              \
      const device itype* bias [[buffer(2)]],                                \
      device itype* out_vec [[buffer(3)]],                                   \
      const constant int& in_vec_size [[buffer(4)]],                         \
      const constant int& out_vec_size [[buffer(5)]],                        \
      const constant int& marix_ld [[buffer(6)]],                            \
      const constant float& alpha [[buffer(7)]],                             \
      const constant float& beta [[buffer(8)]],                              \
      const constant int& batch_ndim [[buffer(9)]],                          \
      const constant int* batch_shape [[buffer(10)]],                        \
      const constant size_t* vector_batch_stride [[buffer(11)]],             \
      const constant size_t* matrix_batch_stride [[buffer(12)]],             \
      const constant size_t* bias_batch_stride [[buffer(13)]],               \
      const constant int& bias_stride [[buffer(14)]],                        \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_gemv(name, itype, bm, bn, tm, tn)        \
  instantiate_gemv_helper(name, itype, bm, bn, tm, tn, 0, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, tm, tn, 0, 1) \
  instantiate_gemv_helper(name, itype, bm, bn, tm, tn, 1, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, tm, tn, 1, 1) // clang-format on

// clang-format off
#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 4, 32, 1, 4) \
  instantiate_gemv(name, itype, 4, 32, 4, 4) \
  instantiate_gemv(name, itype, 8, 32, 4, 4) // clang-format on

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM* BN)]] void gemv_bs(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant size_t* index_batch_strides [[buffer(11)]],
    const constant int& vector_batch_ndim [[buffer(12)]],
    const constant int* vector_batch_shape [[buffer(13)]],
    const constant size_t* vector_batch_stride [[buffer(14)]],
    const constant int& matrix_batch_ndim [[buffer(15)]],
    const constant int* matrix_batch_shape [[buffer(16)]],
    const constant size_t* matrix_batch_stride [[buffer(17)]],
    const constant uint32_t* vec_indices [[buffer(18)]],
    const constant uint32_t* mat_indices [[buffer(19)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVKernel<T, BM, BN, TM, TN, false>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  uint32_t indx_vec;
  uint32_t indx_mat;

  // Update batch offsets
  if (batch_ndim > 1) {
    const constant size_t* veci_bstrides = index_batch_strides;
    const constant size_t* mati_bstrides = index_batch_strides + batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, veci_bstrides, mati_bstrides, batch_ndim);

    indx_vec = vec_indices[batch_offsets.x];
    indx_mat = mat_indices[batch_offsets.y];

  } else {
    indx_vec = vec_indices[index_batch_strides[0] * tid.z];
    indx_mat = mat_indices[index_batch_strides[batch_ndim] * tid.z];
  }

  if (vector_batch_ndim > 1) {
    in_vec += elem_to_loc(
        indx_vec, vector_batch_shape, vector_batch_stride, vector_batch_ndim);
  } else {
    in_vec += indx_vec * vector_batch_stride[0];
  }

  if (matrix_batch_ndim > 1) {
    mat += elem_to_loc(
        indx_mat, matrix_batch_shape, matrix_batch_stride, matrix_batch_ndim);
  } else {
    mat += indx_mat * matrix_batch_stride[0];
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      batch_ndim, // Not used
      tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

#define instantiate_gemv_bs_helper(nm, itype, bm, bn, tm, tn)       \
  template [[host_name("gemv_bs_" #nm "_bm" #bm "_bn" #bn "_tm" #tm \
                       "_tn" #tn)]] [[kernel]] void                 \
  gemv_bs<itype, bm, bn, tm, tn>(                                   \
      const device itype* mat [[buffer(0)]],                        \
      const device itype* in_vec [[buffer(1)]],                     \
      const device itype* bias [[buffer(2)]],                       \
      device itype* out_vec [[buffer(3)]],                          \
      const constant int& in_vec_size [[buffer(4)]],                \
      const constant int& out_vec_size [[buffer(5)]],               \
      const constant int& marix_ld [[buffer(6)]],                   \
      const constant float& alpha [[buffer(7)]],                    \
      const constant float& beta [[buffer(8)]],                     \
      const constant int& batch_ndim [[buffer(9)]],                 \
      const constant int* batch_shape [[buffer(10)]],               \
      const constant size_t* index_batch_strides [[buffer(11)]],    \
      const constant int& vector_batch_ndim [[buffer(12)]],         \
      const constant int* vector_batch_shape [[buffer(13)]],        \
      const constant size_t* vector_batch_stride [[buffer(14)]],    \
      const constant int& matrix_batch_ndim [[buffer(15)]],         \
      const constant int* matrix_batch_shape [[buffer(16)]],        \
      const constant size_t* matrix_batch_stride [[buffer(17)]],    \
      const constant uint32_t* vec_indices [[buffer(18)]],          \
      const constant uint32_t* mat_indices [[buffer(19)]],          \
      uint3 tid [[threadgroup_position_in_grid]],                   \
      uint3 lid [[thread_position_in_threadgroup]],                 \
      uint simd_gid [[simdgroup_index_in_threadgroup]],             \
      uint simd_lid [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_gemv_bs_blocks(name, itype)        \
  instantiate_gemv_bs_helper(name, itype, 4, 32, 1, 4) \
  instantiate_gemv_bs_helper(name, itype, 4, 32, 4, 4) \
  instantiate_gemv_bs_helper(name, itype, 8, 32, 4, 4) // clang-format on

instantiate_gemv_bs_blocks(float32, float);
instantiate_gemv_bs_blocks(float16, half);
instantiate_gemv_bs_blocks(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
[[kernel, max_total_threads_per_threadgroup(BM* BN)]] void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant size_t* vector_batch_stride [[buffer(11)]],
    const constant size_t* matrix_batch_stride [[buffer(12)]],
    const constant size_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVTKernel<T, BM, BN, TM, TN, kDoAxpby>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (kDoAxpby) {
      bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (kDoAxpby) {
      bias += tid.z * bias_batch_stride[0];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      bias_stride,
      tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

#define instantiate_gemv_t_helper(name, itype, bm, bn, tm, tn, nc, axpby)      \
  template [[host_name("gemv_t_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn \
                       "_nc" #nc "_axpby" #axpby)]] [[kernel]] void            \
  gemv_t<itype, bm, bn, tm, tn, nc, axpby>(                                    \
      const device itype* mat [[buffer(0)]],                                   \
      const device itype* in_vec [[buffer(1)]],                                \
      const device itype* bias [[buffer(2)]],                                  \
      device itype* out_vec [[buffer(3)]],                                     \
      const constant int& in_vec_size [[buffer(4)]],                           \
      const constant int& out_vec_size [[buffer(5)]],                          \
      const constant int& marix_ld [[buffer(6)]],                              \
      const constant float& alpha [[buffer(7)]],                               \
      const constant float& beta [[buffer(8)]],                                \
      const constant int& batch_ndim [[buffer(9)]],                            \
      const constant int* batch_shape [[buffer(10)]],                          \
      const constant size_t* vector_batch_stride [[buffer(11)]],               \
      const constant size_t* matrix_batch_stride [[buffer(12)]],               \
      const constant size_t* bias_batch_stride [[buffer(13)]],                 \
      const constant int& bias_stride [[buffer(14)]],                          \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_gemv_t(name, itype, bm, bn, tm, tn)        \
  instantiate_gemv_t_helper(name, itype, bm, bn, tm, tn, 0, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, tm, tn, 0, 1) \
  instantiate_gemv_t_helper(name, itype, bm, bn, tm, tn, 1, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, tm, tn, 1, 1) // clang-format on

// clang-format off
#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 8, 8, 4, 1)  \
  instantiate_gemv_t(name, itype, 8, 8, 4, 4)  \
  instantiate_gemv_t(name, itype, 8, 16, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 32, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 64, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 128, 4, 4) // clang-format on

// clang-format off
instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t); // clang-format on

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM* BN)]] void gemv_t_bs(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant size_t* index_batch_strides [[buffer(11)]],
    const constant int& vector_batch_ndim [[buffer(12)]],
    const constant int* vector_batch_shape [[buffer(13)]],
    const constant size_t* vector_batch_stride [[buffer(14)]],
    const constant int& matrix_batch_ndim [[buffer(15)]],
    const constant int* matrix_batch_shape [[buffer(16)]],
    const constant size_t* matrix_batch_stride [[buffer(17)]],
    const constant uint32_t* vec_indices [[buffer(18)]],
    const constant uint32_t* mat_indices [[buffer(19)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVTKernel<T, BM, BN, TM, TN, false>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  uint32_t indx_vec;
  uint32_t indx_mat;

  // Update batch offsets
  if (batch_ndim > 1) {
    const constant size_t* veci_bstrides = index_batch_strides;
    const constant size_t* mati_bstrides = index_batch_strides + batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, veci_bstrides, mati_bstrides, batch_ndim);

    indx_vec = vec_indices[batch_offsets.x];
    indx_mat = mat_indices[batch_offsets.y];

  } else {
    indx_vec = vec_indices[index_batch_strides[0] * tid.z];
    indx_mat = mat_indices[index_batch_strides[batch_ndim] * tid.z];
  }

  if (vector_batch_ndim > 1) {
    in_vec += elem_to_loc(
        indx_vec, vector_batch_shape, vector_batch_stride, vector_batch_ndim);
  } else {
    in_vec += indx_vec * vector_batch_stride[0];
  }

  if (matrix_batch_ndim > 1) {
    mat += elem_to_loc(
        indx_mat, matrix_batch_shape, matrix_batch_stride, matrix_batch_ndim);
  } else {
    mat += indx_mat * matrix_batch_stride[0];
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      batch_ndim, // Not used,
      tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

#define instantiate_gemv_t_bs_helper(nm, itype, bm, bn, tm, tn)       \
  template [[host_name("gemv_t_bs_" #nm "_bm" #bm "_bn" #bn "_tm" #tm \
                       "_tn" #tn)]] [[kernel]] void                   \
  gemv_t_bs<itype, bm, bn, tm, tn>(                                   \
      const device itype* mat [[buffer(0)]],                          \
      const device itype* in_vec [[buffer(1)]],                       \
      const device itype* bias [[buffer(2)]],                         \
      device itype* out_vec [[buffer(3)]],                            \
      const constant int& in_vec_size [[buffer(4)]],                  \
      const constant int& out_vec_size [[buffer(5)]],                 \
      const constant int& marix_ld [[buffer(6)]],                     \
      const constant float& alpha [[buffer(7)]],                      \
      const constant float& beta [[buffer(8)]],                       \
      const constant int& batch_ndim [[buffer(9)]],                   \
      const constant int* batch_shape [[buffer(10)]],                 \
      const constant size_t* index_batch_strides [[buffer(11)]],      \
      const constant int& vector_batch_ndim [[buffer(12)]],           \
      const constant int* vector_batch_shape [[buffer(13)]],          \
      const constant size_t* vector_batch_stride [[buffer(14)]],      \
      const constant int& matrix_batch_ndim [[buffer(15)]],           \
      const constant int* matrix_batch_shape [[buffer(16)]],          \
      const constant size_t* matrix_batch_stride [[buffer(17)]],      \
      const constant uint32_t* vec_indices [[buffer(18)]],            \
      const constant uint32_t* mat_indices [[buffer(19)]],            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint3 lid [[thread_position_in_threadgroup]],                   \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_gemv_t_bs_blocks(name, itype) \
  instantiate_gemv_t_bs_helper(name, itype, 8, 8, 4, 1)  \
  instantiate_gemv_t_bs_helper(name, itype, 8, 8, 4, 4)  \
  instantiate_gemv_t_bs_helper(name, itype, 8, 16, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 8, 32, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 8, 64, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 8, 128, 4, 4) // clang-format on

// clang-format off
instantiate_gemv_t_bs_blocks(float32, float);
instantiate_gemv_t_bs_blocks(float16, half);
instantiate_gemv_t_bs_blocks(bfloat16, bfloat16_t); // clang-format on