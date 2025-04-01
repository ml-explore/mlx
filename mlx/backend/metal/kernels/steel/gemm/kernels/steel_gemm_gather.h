// Copyright © 2024 Apple Inc.

using namespace mlx::steel;

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

  // Calculate the number of unique matmuls
  uint32_t unique_indices[4];
  short offsets[5];
  unique_indices[0] = rhs_indices[c_row];
  offsets[0] = 0;
  int j = 0;
  for (int i = 1; i < tgp_bm; i++) {
    if (rhs_indices[c_row + i] != unique_indices[j]) {
      j++;
      offsets[j] = i;
      unique_indices[j] = rhs_indices[c_row + i];
    }
  }
  int n_unique = j + 1;
  offsets[n_unique] = tgp_bm;

  threadgroup_barrier(mem_flags::mem_none);

  for (int n = 0; n < n_unique; n++) {
    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(
        B + unique_indices[n] * params->batch_stride_b,
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
      if (offsets[n + 1] - offsets[n] == BM) {
        mma_op.store_result(C, params->ldd);
      } else {
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offsets[n]), short2(BN, offsets[n + 1]));
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
        if (offsets[n + 1] - offsets[n] == BM) {
          mma_op.store_result(C, params->ldd);
        } else {
          mma_op.store_result_slice(
              C,
              params->ldd,
              short2(0, offsets[n]),
              short2(BN, offsets[n + 1]));
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
            C, params->ldd, short2(0, offsets[n]), short2(BN, offsets[n + 1]));
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
            C,
            params->ldd,
            short2(0, offsets[n]),
            short2(tgp_bn, offsets[n + 1]));
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
            C,
            params->ldd,
            short2(0, offsets[n]),
            short2(tgp_bn, offsets[n + 1]));
      }
    }
  }
}
