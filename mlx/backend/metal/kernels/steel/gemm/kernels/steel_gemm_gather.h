// Copyright © 2024 Apple Inc.

using namespace mlx::steel;

constant bool has_batch [[function_constant(10)]];
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

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
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gather_mm_rhs(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* rhs_indices [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
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

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  // Find the block in A, B, C
  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = align_N ? BN : short(min(BN, params->N - c_col));

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = rhs_indices[c_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (rhs_indices[c_row + n] != index) {
        offset_next = n;
        index_next = rhs_indices[c_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(
        B + index * params->batch_stride_b,
        params->ldb,
        Bs,
        simd_group_id,
        simd_lane_id);

    // Prepare iterations
    const int gemm_k_iterations = params->gemm_k_iterations_aligned;

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

    // Matrix level aligned never check
    if (align_M && align_N) {
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

      // Store results to device memory
      if (offset_next - offset == BM) {
        mma_op.store_result(C, params->ldd);
      } else {
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      const short lbk = 0;

      // Tile aligned don't check
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<true, true, true>{});
        if (offset_next - offset == BM) {
          mma_op.store_result(C, params->ldd);
        } else {
          mma_op.store_result_slice(
              C, params->ldd, short2(0, offset), short2(BN, offset_next));
        }
      }

      // Tile partially aligned check rows
      else if (align_N || tgp_bn == BN) {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<false, true, true>{});
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(BN, offset_next));
      }

      // Tile partially aligned check cols
      else if (align_M || tgp_bm == BM) {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<true, false, true>{});
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(tgp_bn, offset_next));
      }

      // Nothing aligned so check both rows and cols
      else {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<false, false, true>{});
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(tgp_bn, offset_next));
      }
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
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gather_mm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* lhs_indices [[buffer(2)]],
    const device uint32_t* rhs_indices [[buffer(3)]],
    device T* C [[buffer(4)]],
    const constant GEMMParams* params [[buffer(5)]],
    const constant int* indices_shape [[buffer(6)]],
    const constant int64_t* lhs_strides [[buffer(7)]],
    const constant int64_t* rhs_strides [[buffer(8)]],
    const constant int& batch_ndim_a [[buffer(9)]],
    const constant int* batch_shape_a [[buffer(10)]],
    const constant int64_t* batch_strides_a [[buffer(11)]],
    const constant int& batch_ndim_b [[buffer(12)]],
    const constant int* batch_shape_b [[buffer(13)]],
    const constant int64_t* batch_strides_b [[buffer(14)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
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

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  // Move A and B to the locations pointed by lhs_indices and rhs_indices.
  uint32_t indx_A, indx_B;
  if (has_batch) {
    ulong2 indices_offsets = elem_to_loc_broadcast(
        tid.z, indices_shape, lhs_strides, rhs_strides, params->batch_ndim);
    indx_A = lhs_indices[indices_offsets.x];
    indx_B = rhs_indices[indices_offsets.y];
  } else {
    indx_A = lhs_indices[params->batch_stride_a * tid.z];
    indx_B = rhs_indices[params->batch_stride_b * tid.z];
  }
  A += elem_to_loc(indx_A, batch_shape_a, batch_strides_a, batch_ndim_a);
  B += elem_to_loc(indx_B, batch_shape_b, batch_strides_b, batch_ndim_b);
  C += params->batch_stride_d * tid.z;

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  // Just make sure everybody's finished with the indexing math above.
  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

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

  // Matrix level aligned never check
  if (align_M && align_N) {
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

    // Store results to device memory
    mma_op.store_result(C, params->ldd);
  } else {
    const short lbk = 0;

    // Tile aligned don't check
    if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<true, true, true>{});
      mma_op.store_result(C, params->ldd);
    }

    // Tile partially aligned check rows
    else if (align_N || tgp_bn == BN) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<false, true, true>{});
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }

    // Tile partially aligned check cols
    else if (align_M || tgp_bm == BM) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<true, false, true>{});
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }

    // Nothing aligned so check both rows and cols
    else {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<false, false, true>{});
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}
