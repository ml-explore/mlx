// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/steel/defines.h"
using namespace metal;
using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

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

template <typename OutT, typename InT = OutT>
struct ScaleOp {
  OutT scale;

  METAL_FUNC OutT apply(InT x) const {
    return static_cast<OutT>(x) * scale;
  }
};

typedef struct _NoMask nomask_t;

template <
    typename T,
    typename out_mask_t,
    typename op_mask_t,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
block_masked_gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant int64_t* batch_strides [[buffer(7)]],
    const device out_mask_t* out_mask [[buffer(10)]],
    const device op_mask_t* lhs_mask [[buffer(11)]],
    const device op_mask_t* rhs_mask [[buffer(12)]],
    const constant int* mask_strides [[buffer(13)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // Appease the compiler
  (void)lid;

  static_assert(
      BM == BN,
      "block_masked_gemm must have the same block M and block N size");
  static_assert(BM % BK == 0, "block_masked_gemm must have BM % BK == 0");

  constexpr bool has_operand_mask = !metal::is_same_v<op_mask_t, nomask_t>;
  constexpr bool has_output_mask = !metal::is_same_v<out_mask_t, nomask_t>;

  constexpr bool has_mul_operand_mask =
      has_operand_mask && !metal::is_same_v<op_mask_t, bool>;
  constexpr bool has_mul_output_mask =
      has_output_mask && !metal::is_same_v<out_mask_t, bool>;

  constexpr short k_mask_factor = short(BM / BK);

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      MN_aligned,
      K_aligned>;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const constant auto* mask_batch_strides =
      batch_strides + 2 * params->batch_ndim;

  if (params->batch_ndim > 1) {
    if (has_output_mask) {
      out_mask += elem_to_loc(
          tid.z, batch_shape, mask_batch_strides, params->batch_ndim);

      mask_batch_strides += params->batch_ndim;
    }

    if (has_operand_mask) {
      const constant auto* mask_strides_lhs = mask_batch_strides;
      const constant auto* mask_strides_rhs =
          mask_strides_lhs + params->batch_ndim;

      ulong2 batch_offsets = elem_to_loc_broadcast(
          tid.z,
          batch_shape,
          mask_strides_lhs,
          mask_strides_rhs,
          params->batch_ndim);

      lhs_mask += batch_offsets.x;
      rhs_mask += batch_offsets.y;
    }
  } else {
    if (has_output_mask) {
      out_mask += tid.z * mask_batch_strides[0];
      mask_batch_strides += params->batch_ndim;
    }

    if (has_operand_mask) {
      lhs_mask += tid.z * mask_batch_strides[0];
      rhs_mask += tid.z * mask_batch_strides[params->batch_ndim];
    }
  }

  // Adjust for batch
  if (params->batch_ndim > 1) {
    const constant auto* A_bstrides = batch_strides;
    const constant auto* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    A += batch_offsets.x;
    B += batch_offsets.y;

  } else {
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;
  }

  D += params->batch_stride_d * tid.z;

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  D += c_row_long * params->ldd + c_col_long;

  const constant int* out_mask_strides = mask_strides;
  const constant int* lhs_mask_strides =
      mask_strides + (has_output_mask ? 2 : 0);
  const constant int* rhs_mask_strides =
      lhs_mask_strides + (has_operand_mask ? 2 : 0);

  const int out_mask_offset = !has_output_mask
      ? 0
      : tid_y * out_mask_strides[1] + tid_x * out_mask_strides[0];
  int lhs_mask_offset = !has_operand_mask ? 0 : tid_y * lhs_mask_strides[1];
  int rhs_mask_offset = !has_operand_mask ? 0 : tid_x * rhs_mask_strides[0];
  const int lhs_mask_step = !has_operand_mask ? 0 : lhs_mask_strides[0];
  const int rhs_mask_step = !has_operand_mask ? 0 : rhs_mask_strides[1];
  short k_factor_cnt = k_mask_factor;

  ScaleOp<float> out_mask_op;
  ScaleOp<T> lhs_mask_op;
  ScaleOp<T> rhs_mask_op;

  if (has_output_mask) {
    auto mask_out = out_mask[out_mask_offset];

    if (has_mul_output_mask) {
      out_mask_op.scale = float(mask_out);
    }

    // Write zeros and return
    if (!mask_out) {
      constexpr short tgp_size = WM * WN * 32;
      constexpr short vec_size = 4;

      // Tile threads in threadgroup
      constexpr short TN = BN / vec_size;
      constexpr short TM = tgp_size / TN;

      const short thread_idx = simd_group_id * 32 + simd_lane_id;
      const short bi = thread_idx / TN;
      const short bj = vec_size * (thread_idx % TN);

      D += bi * params->ldd + bj;

      short tgp_bm = min(BM, params->M - c_row);
      short tgp_bn = min(BN, params->N - c_col);

      if (MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
        for (short ti = 0; ti < BM; ti += TM) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            D[ti * params->ldd + j] = T(0.);
          }
        }
      } else {
        short jmax = tgp_bn - bj;
        jmax = jmax < vec_size ? jmax : vec_size;
        for (short ti = 0; (bi + ti) < tgp_bm; ti += TM) {
          for (short j = 0; j < jmax; j++) {
            D[ti * params->ldd + j] = T(0.);
          }
        }
      }

      return;
    }
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Prepare threadgroup mma operation
  thread typename gemm_kernel::mma_t mma_op(simd_group_id, simd_lane_id);

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  // Prepare threadgroup loading operations
  thread typename gemm_kernel::loader_a_t loader_a(
      A, params->lda, As, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::loader_b_t loader_b(
      B, params->ldb, Bs, simd_group_id, simd_lane_id);

  // Prepare threadgroup bounds
  const short tgp_bm =
      MN_aligned ? short(BM) : short(min(BM, params->M - c_row));
  const short tgp_bn =
      MN_aligned ? short(BN) : short(min(BN, params->N - c_col));

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  ///////////////////////////////////////////////////////////////////////////////
  // Do unaligned K iterations first
  if (!K_aligned) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int mask_idx_last = k_last / BM;

    if (!has_operand_mask ||
        (bool(lhs_mask[lhs_mask_offset + mask_idx_last * lhs_mask_step]) &&
         bool(rhs_mask[rhs_mask_offset + mask_idx_last * rhs_mask_step]))) {
      if (has_mul_operand_mask) {
        lhs_mask_op.scale =
            lhs_mask[lhs_mask_offset + mask_idx_last * lhs_mask_step];
        rhs_mask_op.scale =
            rhs_mask[rhs_mask_offset + mask_idx_last * rhs_mask_step];
      }

      // Move loader source ahead to end
      const int k_remain = params->K - k_last;
      const size_t k_jump_a =
          transpose_a ? params->lda * size_t(k_last) : size_t(k_last);
      const size_t k_jump_b =
          transpose_b ? size_t(k_last) : params->ldb * size_t(k_last);

      loader_a.src += k_jump_a;
      loader_b.src += k_jump_b;

      // Load tile
      const short2 tile_dims_A =
          transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
      const short2 tile_dims_B =
          transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

      loader_a.load_safe(tile_dims_A);
      loader_b.load_safe(tile_dims_B);

      if (has_mul_operand_mask) {
        loader_a.apply_inplace_op(lhs_mask_op);
        loader_b.apply_inplace_op(rhs_mask_op);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Do matmul
      mma_op.mma(As, Bs);

      // Reset source back to start
      loader_a.src -= k_jump_a;
      loader_b.src -= k_jump_b;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  // MNK aligned loop
  if (MN_aligned) {
    for (; gemm_k_iterations > 0; gemm_k_iterations--) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (!has_operand_mask ||
          (bool(lhs_mask[lhs_mask_offset]) &&
           bool(rhs_mask[rhs_mask_offset]))) {
        if (has_mul_operand_mask) {
          lhs_mask_op.scale = lhs_mask[lhs_mask_offset];
          rhs_mask_op.scale = rhs_mask[rhs_mask_offset];
        }

        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        if (has_mul_operand_mask) {
          loader_a.apply_inplace_op(lhs_mask_op);
          loader_b.apply_inplace_op(rhs_mask_op);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);
      }

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();

      k_factor_cnt--;
      lhs_mask_offset += k_factor_cnt == 0 ? lhs_mask_step : 0;
      rhs_mask_offset += k_factor_cnt == 0 ? rhs_mask_step : 0;
      k_factor_cnt = k_factor_cnt == 0 ? k_mask_factor : k_factor_cnt;
    }

    if (has_mul_output_mask) {
      mma_op.apply_epilogue(out_mask_op);
    }

    // Store results to device memory
    mma_op.store_result(D, params->ldd);
    return;

  }
  ///////////////////////////////////////////////////////////////////////////////
  // MN unaligned loop
  else {
    const bool M_aligned = (tgp_bm == BM);
    const bool N_aligned = (tgp_bn == BN);

    const short2 tile_dims_A =
        transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
    const short2 tile_dims_B =
        transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);

    for (; gemm_k_iterations > 0; gemm_k_iterations--) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (!has_operand_mask ||
          (bool(lhs_mask[lhs_mask_offset]) &&
           bool(rhs_mask[rhs_mask_offset]))) {
        if (has_mul_operand_mask) {
          lhs_mask_op.scale = lhs_mask[lhs_mask_offset];
          rhs_mask_op.scale = rhs_mask[rhs_mask_offset];
        }

        // Load elements into threadgroup
        if (M_aligned) {
          loader_a.load_unsafe();
        } else {
          loader_a.load_safe(tile_dims_A);
        }

        if (N_aligned) {
          loader_b.load_unsafe();
        } else {
          loader_b.load_safe(tile_dims_B);
        }

        if (has_mul_operand_mask) {
          loader_a.apply_inplace_op(lhs_mask_op);
          loader_b.apply_inplace_op(rhs_mask_op);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);
      }

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();

      k_factor_cnt--;
      lhs_mask_offset += k_factor_cnt == 0 ? lhs_mask_step : 0;
      rhs_mask_offset += k_factor_cnt == 0 ? rhs_mask_step : 0;
      k_factor_cnt = k_factor_cnt == 0 ? k_mask_factor : k_factor_cnt;
    }

    if (has_mul_output_mask) {
      mma_op.apply_epilogue(out_mask_op);
    }

    if (M_aligned && N_aligned) {
      mma_op.store_result(D, params->ldd);
    } else {
      mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned,
    bool has_operand_mask = false>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
block_masked_gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant int64_t* batch_strides [[buffer(7)]],
    const device bool* out_mask [[buffer(10)]],
    const device bool* lhs_mask [[buffer(11)]],
    const device bool* rhs_mask [[buffer(12)]],
    const constant int* mask_strides [[buffer(13)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // Appease the compiler
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      MN_aligned,
      K_aligned>;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  if (params->batch_ndim > 1) {
    const constant auto* mask_batch_strides =
        batch_strides + 2 * params->batch_ndim;
    out_mask +=
        elem_to_loc(tid.z, batch_shape, mask_batch_strides, params->batch_ndim);

    if (has_operand_mask) {
      const constant auto* mask_strides_lhs =
          mask_batch_strides + params->batch_ndim;
      const constant auto* mask_strides_rhs =
          mask_strides_lhs + params->batch_ndim;

      ulong2 batch_offsets = elem_to_loc_broadcast(
          tid.z,
          batch_shape,
          mask_strides_lhs,
          mask_strides_rhs,
          params->batch_ndim);

      lhs_mask += batch_offsets.x;
      rhs_mask += batch_offsets.y;
    }
  } else {
    out_mask += tid.z * batch_strides[2 * params->batch_ndim];
    if (has_operand_mask) {
      lhs_mask += tid.z * batch_strides[3 * params->batch_ndim];
      rhs_mask += tid.z * batch_strides[4 * params->batch_ndim];
    }
  }

  // Adjust for batch
  if (params->batch_ndim > 1) {
    const constant auto* A_bstrides = batch_strides;
    const constant auto* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    A += batch_offsets.x;
    B += batch_offsets.y;

  } else {
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;
  }

  D += params->batch_stride_d * tid.z;

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  D += c_row_long * params->ldd + c_col_long;

  bool mask_out = out_mask[tid_y * mask_strides[1] + tid_x * mask_strides[0]];

  // Write zeros and return
  if (!mask_out) {
    constexpr short tgp_size = WM * WN * 32;
    constexpr short vec_size = 4;

    // Tile threads in threadgroup
    constexpr short TN = BN / vec_size;
    constexpr short TM = tgp_size / TN;

    const short thread_idx = simd_group_id * 32 + simd_lane_id;
    const short bi = thread_idx / TN;
    const short bj = vec_size * (thread_idx % TN);

    D += bi * params->ldd + bj;

    short tgp_bm = min(BM, params->M - c_row);
    short tgp_bn = min(BN, params->N - c_col);

    if (MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
      for (short ti = 0; ti < BM; ti += TM) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          D[ti * params->ldd + j] = T(0.);
        }
      }
    } else {
      short jmax = tgp_bn - bj;
      jmax = jmax < vec_size ? jmax : vec_size;
      for (short ti = 0; (bi + ti) < tgp_bm; ti += TM) {
        for (short j = 0; j < jmax; j++) {
          D[ti * params->ldd + j] = T(0.);
        }
      }
    }

    return;
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Prepare threadgroup mma operation
  thread typename gemm_kernel::mma_t mma_op(simd_group_id, simd_lane_id);

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  // Prepare threadgroup loading operations
  thread typename gemm_kernel::loader_a_t loader_a(
      A, params->lda, As, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::loader_b_t loader_b(
      B, params->ldb, Bs, simd_group_id, simd_lane_id);

  ///////////////////////////////////////////////////////////////////////////////
  // MNK aligned loop
  if (MN_aligned) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (!has_operand_mask ||
          (lhs_mask
               [tid_y * mask_strides[3] + ((k * BK) / BM) * mask_strides[2]] &&
           rhs_mask
               [((k * BK) / BM) * mask_strides[5] + tid_x * mask_strides[4]])) {
        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);
      }

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Loop tail
    if (!K_aligned) {
      if (!has_operand_mask ||
          (lhs_mask
               [tid_y * mask_strides[3] + (params->K / BM) * mask_strides[2]] &&
           rhs_mask
               [(params->K / BM) * mask_strides[5] +
                tid_x * mask_strides[4]])) {
        int lbk = params->K - params->gemm_k_iterations_aligned * BK;
        short2 tile_dims_A = transpose_a ? short2(BM, lbk) : short2(lbk, BM);
        short2 tile_dims_B = transpose_b ? short2(lbk, BN) : short2(BN, lbk);

        loader_a.load_safe(tile_dims_A);
        loader_b.load_safe(tile_dims_B);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(As, Bs);
      }
    }

    // Store results to device memory
    mma_op.store_result(D, params->ldd);
    return;

  }
  ///////////////////////////////////////////////////////////////////////////////
  // MN unaligned loop
  else { // Loop over K - unaligned case
    short tgp_bm = min(BM, params->M - c_row);
    short tgp_bn = min(BN, params->N - c_col);
    short lbk = params->K - params->gemm_k_iterations_aligned * BK;

    bool M_aligned = (tgp_bm == BM);
    bool N_aligned = (tgp_bn == BN);

    short2 tile_dims_A = transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
    short2 tile_dims_B = transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);

    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (!has_operand_mask ||
          (lhs_mask
               [tid_y * mask_strides[3] + ((k * BK) / BM) * mask_strides[2]] &&
           rhs_mask
               [((k * BK) / BM) * mask_strides[5] + tid_x * mask_strides[4]])) {
        // Load elements into threadgroup
        if (M_aligned) {
          loader_a.load_unsafe();
        } else {
          loader_a.load_safe(tile_dims_A);
        }

        if (N_aligned) {
          loader_b.load_unsafe();
        } else {
          loader_b.load_safe(tile_dims_B);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);
      }

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }

    if (!K_aligned) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (!has_operand_mask ||
          (lhs_mask
               [tid_y * mask_strides[3] + (params->K / BM) * mask_strides[2]] &&
           rhs_mask
               [(params->K / BM) * mask_strides[5] +
                tid_x * mask_strides[4]])) {
        short2 tile_dims_A_last =
            transpose_a ? short2(tgp_bm, lbk) : short2(lbk, tgp_bm);
        short2 tile_dims_B_last =
            transpose_b ? short2(lbk, tgp_bn) : short2(tgp_bn, lbk);

        loader_a.load_safe(tile_dims_A_last);
        loader_b.load_safe(tile_dims_B_last);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(As, Bs);
      }
    }

    if (M_aligned && N_aligned) {
      mma_op.store_result(D, params->ldd);
    } else {
      mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}
