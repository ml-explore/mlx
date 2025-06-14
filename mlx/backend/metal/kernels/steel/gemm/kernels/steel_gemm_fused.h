// Copyright © 2024 Apple Inc.

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

// clang-format off
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2), function_constant(use_out_source)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
    const constant int64_t* batch_strides [[buffer(7), function_constant(has_batch)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on
  // Pacifying compiler
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
      true,
      true,
      AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  // Find block
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  // Exit early if out of bounds
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Adjust for batch
  if (has_batch) {
    const constant auto* A_bstrides = batch_strides;
    const constant auto* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    A += batch_offsets.x;
    B += batch_offsets.y;

    if (use_out_source) {
      const constant auto* C_bstrides = B_bstrides + params->batch_ndim;
      C += elem_to_loc(tid.z, batch_shape, C_bstrides, params->batch_ndim);
    }
  } else {
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;

    if (use_out_source) {
      C += addmm_params->batch_stride_c * tid.z;
    }
  }

  D += params->batch_stride_d * tid.z;

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  D += c_row_long * params->ldd + c_col_long;

  if (use_out_source) {
    C += c_row_long * addmm_params->ldc + c_col_long * addmm_params->fdc;
  }

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = align_N ? BN : short(min(BN, params->N - c_col));

  // Prepare iterations
  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Do unaligned K iterations first
  if (!align_K) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int k_remain = params->K - k_last;
    const size_t k_jump_a =
        transpose_a ? params->lda * size_t(k_last) : size_t(k_last);
    const size_t k_jump_b =
        transpose_b ? size_t(k_last) : params->ldb * size_t(k_last);

    // Move loader source ahead to end
    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;

    // Load tile
    const short2 tile_dims_A =
        transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
    const short2 tile_dims_B =
        transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do matmul
    mma_op.mma(As, Bs);

    // Reset source back to start
    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  const TransformAdd<AccumType, AccumType> epilogue_op_add(
      addmm_params->alpha, addmm_params->beta);
  const TransformAxpby<AccumType, AccumType> epilogue_op_axpby(
      addmm_params->alpha, addmm_params->beta);

  ///////////////////////////////////////////////////////////////////////////////
  // MNK aligned loop
  if (align_M && align_N) {
    // Do gemm
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Load elements into threadgroup
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Do epilogue
    if (use_out_source) {
      if (do_axpby) {
        mma_op.apply_epilogue(
            C, addmm_params->ldc, addmm_params->fdc, epilogue_op_axpby);
      } else {
        mma_op.apply_epilogue(
            C, addmm_params->ldc, addmm_params->fdc, epilogue_op_add);
      }
    }

    // Store results to device memory
    return mma_op.store_result(D, params->ldd);

  }
  ///////////////////////////////////////////////////////////////////////////////
  // MN unaligned loop
  else { // Loop over K - unaligned case
    const int leftover_bk = 0;

    if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
      // Do gemm
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, true, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue(
              C, addmm_params->ldc, addmm_params->fdc, epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue(
              C, addmm_params->ldc, addmm_params->fdc, epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result(D, params->ldd);

    } else if (align_N || tgp_bn == BN) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, true, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));

    } else if (align_M || tgp_bm == BM) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, false, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));

    } else {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, false, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}
