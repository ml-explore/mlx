// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

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

template <typename OutT, typename InT = OutT>
struct ScaleOp {
  OutT scale;

  METAL_FUNC OutT apply(InT x) const {
    return static_cast<OutT>(x) * scale;
  }
};

template <
    typename T,
    typename out_mask_t,
    typename op_mask_t,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    typename AccT = float>
struct GEMVKernel {
  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  static_assert(
      SN == 8 || SN == 16 || SN == 32,
      "gemv block must have a width of 8, 16, or 32");

  static_assert(blockN >= blockM, "Masked gemv must have blockN >= blockM");

  MLX_MTL_CONST bool has_operand_mask = !metal::is_same_v<op_mask_t, nomask_t>;
  MLX_MTL_CONST bool has_output_mask = !metal::is_same_v<out_mask_t, nomask_t>;

  MLX_MTL_CONST bool has_mul_operand_mask =
      has_operand_mask && !metal::is_same_v<op_mask_t, bool>;
  MLX_MTL_CONST bool has_mul_output_mask =
      has_output_mask && !metal::is_same_v<out_mask_t, bool>;

  // - The matrix of size (M = out_vec_size, K = in_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM rows
  //    and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for
  //    the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated blockM outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN*(blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  template <typename U = T>
  static METAL_FUNC void
  load_unsafe(const device T* src, thread U dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else { // Edgecase
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& matrix_ld [[buffer(6)]],
      const device out_mask_t* out_mask [[buffer(20)]],
      const device op_mask_t* mat_mask [[buffer(21)]],
      const device op_mask_t* vec_mask [[buffer(22)]],
      const constant int* mask_strides [[buffer(23)]],
      threadgroup AccT* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    thread AccT result[TM] = {0};
    thread T inter[TN];
    thread AccT v_coeff[TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position
    int out_row = tid.x * blockM + bm;

    // Exit simdgroup if rows out of bound
    if (out_row >= out_vec_size)
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Prepare mask offsets
    const constant int* out_mask_strides = mask_strides;
    const constant int* mat_mask_strides =
        mask_strides + (has_output_mask ? 2 : 0);
    const constant int* vec_mask_strides =
        mat_mask_strides + (has_operand_mask ? 2 : 0);

    const int m_block_idx = blockN > blockM ? out_row / blockN : int(tid.x);

    const int out_mask_offset =
        !has_output_mask ? 0 : m_block_idx * out_mask_strides[1];

    int mat_mask_offset =
        !has_operand_mask ? 0 : m_block_idx * mat_mask_strides[1];
    int vec_mask_offset = 0;
    const int mat_mask_step = !has_operand_mask ? 0 : mat_mask_strides[0];
    const int vec_mask_step = !has_operand_mask ? 0 : vec_mask_strides[1];

    T out_scale{1};

    // Check output mask
    if (has_output_mask) {
      auto mask_out = out_mask[out_mask_offset];

      // Write zeros and return if mask is 0
      if (!mask_out) {
        if (simdN == 0 && thrN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            out_vec[out_row + tm] = T(0.);
          }
        }

        return;
      }

      // Store scalar if multiplicative mask
      if (has_mul_output_mask) {
        out_scale = T(mask_out);
      }
    }

    // Advance matrix
    mat += out_row * matrix_ld;

    // Prepare for loop
    constexpr const uniform<int> loop_stride = make_uniform(blockN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Loop over in_vec in blocks of blockN
    for (int i = 0; i < n_iter; ++i) {
      if (!has_operand_mask ||
          (bool(mat_mask[mat_mask_offset]) &&
           bool(vec_mask[vec_mask_offset]))) {
        T block_scale{1};
        if (has_mul_operand_mask) {
          block_scale =
              T(mat_mask[mat_mask_offset]) * T(vec_mask[vec_mask_offset]);
        }

        load_unsafe<AccT>(in_vec, v_coeff, bn);

        // Apply scale
        if (has_mul_operand_mask) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            v_coeff[tn] *= block_scale;
          }
        }

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
      }

      bn += blockN;
      mat_mask_offset += mat_mask_step;
      vec_mask_offset += vec_mask_step;
    }

    if (leftover > 0) {
      if (!has_operand_mask ||
          (bool(mat_mask[mat_mask_offset]) &&
           bool(vec_mask[vec_mask_offset]))) {
        T block_scale{1};
        if (has_mul_operand_mask) {
          block_scale =
              T(mat_mask[mat_mask_offset]) * T(vec_mask[vec_mask_offset]);
        }

        load_safe<AccT>(in_vec, v_coeff, bn, in_size);

        // Apply scale
        if (has_mul_operand_mask) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            v_coeff[tn] *= block_scale;
          }
        }

        // Per thread work loop
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          // Load for the row
          load_safe(&mat[tm * matrix_ld], inter, bn, in_size);

          // Accumulate results
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tm] += inter[tn] * v_coeff[tn];
          }
        }
      }
    }

    // Apply out scale
    if (has_mul_output_mask) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        result[tm] *= out_scale;
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[tm];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    // Write outputs
    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        out_vec[out_row + tm] = static_cast<T>(result[tm]);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename out_mask_t,
    typename op_mask_t,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    typename AccT = float>
struct GEMVTKernel {
  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  MLX_MTL_CONST bool has_operand_mask = !metal::is_same_v<op_mask_t, nomask_t>;
  MLX_MTL_CONST bool has_output_mask = !metal::is_same_v<out_mask_t, nomask_t>;

  MLX_MTL_CONST bool has_mul_operand_mask =
      has_operand_mask && !metal::is_same_v<op_mask_t, bool>;
  MLX_MTL_CONST bool has_mul_output_mask =
      has_output_mask && !metal::is_same_v<out_mask_t, bool>;

  // - The matrix of size (M = in_vec_size, N = out_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
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

  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? BM*(blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& marix_ld [[buffer(6)]],
      const device out_mask_t* out_mask [[buffer(20)]],
      const device op_mask_t* mat_mask [[buffer(21)]],
      const device op_mask_t* vec_mask [[buffer(22)]],
      const constant int* mask_strides [[buffer(23)]],
      threadgroup AccT* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    AccT result[TN] = {0};
    T inter[TN];
    AccT v_coeff[TM];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = SM * sgM;
    const int simdN = SN * sgN;

    int cm = (simdM + thrM);
    int cn = (simdN + thrN);

    int bm = cm * TM;
    int bn = cn * TN;

    int out_col = tid.x * blockN + bn;

    // Prepare mask offsets
    const constant int* out_mask_strides = mask_strides;
    const constant int* mat_mask_strides =
        out_mask_strides + (has_output_mask ? 2 : 0);
    const constant int* vec_mask_strides =
        mat_mask_strides + (has_operand_mask ? 2 : 0);

    const int n_block_idx = blockM > blockN ? out_col / blockM : int(tid.x);

    const int out_mask_offset =
        !has_output_mask ? 0 : n_block_idx; // * out_mask_strides[0];

    int mat_mask_offset =
        !has_operand_mask ? 0 : n_block_idx * mat_mask_strides[0];
    int vec_mask_offset = 0;
    const int mat_mask_step = !has_operand_mask ? 0 : mat_mask_strides[1];
    const int vec_mask_step = !has_operand_mask ? 0 : vec_mask_strides[0];

    T out_scale{1};

    // Check output mask
    if (has_output_mask) {
      auto mask_out = out_mask[out_mask_offset];

      // Write zeros and return if mask is 0
      if (!mask_out) {
        if (cm == 0 && out_col < out_vec_size) {
          if (out_col + TN <= out_vec_size) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              out_vec[out_col + tn] = T(0.);
            }
          } else {
            for (int tn = 0; tn < TN && (out_col + tn) < out_vec_size; tn++) {
              out_vec[out_col + tn] = T(0.);
            }
          }
        }

        return;
      }

      // Store scalar if multiplicative mask
      if (has_mul_output_mask) {
        out_scale = T(mask_out);
      }
    }

    // Prepare for loop
    constexpr const uniform<int> loop_stride = make_uniform(blockM);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Edgecase handling
    if (out_col < out_vec_size) {
      out_col = (out_col + TN) <= out_vec_size ? out_col : out_vec_size - TN;

      // Per thread accumulation main loop
      for (int i = 0; i < n_iter; ++i) {
        // Adding a threadgroup_barrier improves performance slightly
        // This is possibly it may help exploit cache better
        threadgroup_barrier(mem_flags::mem_none);

        if (!has_operand_mask ||
            (bool(mat_mask[mat_mask_offset]) &&
             bool(vec_mask[vec_mask_offset]))) {
          T block_scale{1};
          if (has_mul_operand_mask) {
            block_scale =
                T(mat_mask[mat_mask_offset]) * T(vec_mask[vec_mask_offset]);
          }

          MLX_MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
          }

          // Apply scale
          if (has_mul_operand_mask) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              v_coeff[tm] *= block_scale;
            }
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
        }

        bm += blockM;
        mat_mask_offset += mat_mask_step;
        vec_mask_offset += vec_mask_step;
      }

      if (leftover > 0) {
        if (!has_operand_mask ||
            (bool(mat_mask[mat_mask_offset]) &&
             bool(vec_mask[vec_mask_offset]))) {
          T block_scale{1};
          if (has_mul_operand_mask) {
            block_scale =
                T(mat_mask[mat_mask_offset]) * T(vec_mask[vec_mask_offset]);
          }

          for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
            v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);

            if (has_mul_operand_mask) {
              v_coeff[tm] *= block_scale;
            }

            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
            }

            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += v_coeff[tm] * inter[tn];
            }
          }
        }
      }
    }

    // Apply out scale
    if (has_mul_output_mask) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        result[tn] *= out_scale;
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          tgp_results[tn] = result[tn];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgm = 1; sgm < BM; sgm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += tgp_results[sgm * (blockN + TN) + tn];
            }
          }
        }
      }
    }

    // Threadgroup accumulation and writing out results
    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        out_vec[out_col + j] = static_cast<T>(result[j]);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename out_mask_t,
    typename op_mask_t,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch> /* Batch ndim > 1 */
[[kernel, max_total_threads_per_threadgroup(BM* BN * 32)]] void gemv_masked(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const device out_mask_t* out_mask [[buffer(20)]],
    const device op_mask_t* mat_mask [[buffer(21)]],
    const device op_mask_t* vec_mask [[buffer(22)]],
    const constant int* mask_strides [[buffer(23)]],
    const constant int64_t* mask_batch_strides [[buffer(24)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel =
      GEMVKernel<T, out_mask_t, op_mask_t, BM, BN, SM, SN, TM, TN>;
  threadgroup float tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  constexpr bool has_operand_mask = !metal::is_same_v<op_mask_t, nomask_t>;
  constexpr bool has_output_mask = !metal::is_same_v<out_mask_t, nomask_t>;

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (has_output_mask) {
      out_mask +=
          elem_to_loc(tid.z, batch_shape, mask_batch_strides, batch_ndim);
      mask_batch_strides += batch_ndim;
    }

    if (has_operand_mask) {
      const constant auto* mask_strides_mat = mask_batch_strides;
      const constant auto* mask_strides_vec = mask_strides_mat + batch_ndim;

      ulong2 batch_offsets = elem_to_loc_broadcast(
          tid.z, batch_shape, mask_strides_mat, mask_strides_vec, batch_ndim);

      mat_mask += batch_offsets.x;
      vec_mask += batch_offsets.y;
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (has_output_mask) {
      out_mask += tid.z * mask_batch_strides[0];
      mask_batch_strides += batch_ndim;
    }

    if (has_operand_mask) {
      mat_mask += tid.z * mask_batch_strides[0];
      vec_mask += tid.z * mask_batch_strides[batch_ndim];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      out_mask,
      mat_mask,
      vec_mask,
      mask_strides,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename out_mask_t,
    typename op_mask_t,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch> /* Batch ndim > 1 */
[[kernel, max_total_threads_per_threadgroup(BM* BN * 32)]] void gemv_t_masked(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const device out_mask_t* out_mask [[buffer(20)]],
    const device op_mask_t* mat_mask [[buffer(21)]],
    const device op_mask_t* vec_mask [[buffer(22)]],
    const constant int* mask_strides [[buffer(23)]],
    const constant int64_t* mask_batch_strides [[buffer(24)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel =
      GEMVTKernel<T, out_mask_t, op_mask_t, BM, BN, SM, SN, TM, TN>;
  threadgroup float tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  constexpr bool has_operand_mask = !metal::is_same_v<op_mask_t, nomask_t>;
  constexpr bool has_output_mask = !metal::is_same_v<out_mask_t, nomask_t>;

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (has_output_mask) {
      out_mask +=
          elem_to_loc(tid.z, batch_shape, mask_batch_strides, batch_ndim);
      mask_batch_strides += batch_ndim;
    }

    if (has_operand_mask) {
      const constant auto* mask_strides_mat = mask_batch_strides;
      const constant auto* mask_strides_vec = mask_strides_mat + batch_ndim;

      ulong2 batch_offsets = elem_to_loc_broadcast(
          tid.z, batch_shape, mask_strides_mat, mask_strides_vec, batch_ndim);

      mat_mask += batch_offsets.x;
      vec_mask += batch_offsets.y;
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (has_output_mask) {
      out_mask += tid.z * mask_batch_strides[0];
      mask_batch_strides += batch_ndim;
    }

    if (has_operand_mask) {
      mat_mask += tid.z * mask_batch_strides[0];
      vec_mask += tid.z * mask_batch_strides[batch_ndim];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      out_mask,
      mat_mask,
      vec_mask,
      mask_strides,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}
