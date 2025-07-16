// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/steel/conv/loaders/loader_general.h"

constant bool align_C [[function_constant(200)]];

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = float,
    typename Epilogue = TransformNone<T, AccumType>>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
implicit_gemm_conv_2d_general(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    const constant MLXConvParams<2>* params [[buffer(3)]],
    const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
    const constant Conv2DGeneralJumpParams* jump_params [[buffer(5)]],
    const constant Conv2DGeneralBaseInfo* base_h [[buffer(6)]],
    const constant Conv2DGeneralBaseInfo* base_w [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);

  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;

  constexpr short tgp_size = WM * WN * 32;

  // Input loader
  using loader_a_t =
      Conv2DInputBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_a>;

  // Weight loader
  using loader_b_t =
      Conv2DWeightBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_b>;

  using mma_t = BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      shape_a_cols,
      shape_b_cols>;

  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
      ((tid.x) & ((1 << gemm_params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;

  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) {
    return;
  }

  const int tid_z = tid.z;

  const int base_oh = tid_z / jump_params->f_out_jump_w;
  const int base_ow = tid_z % jump_params->f_out_jump_w;

  const int base_wh = base_h[base_oh].weight_base;
  const int base_ww = base_w[base_ow].weight_base;

  const int base_wh_size = base_h[base_oh].weight_size;
  const int base_ww_size = base_w[base_ow].weight_size;

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;

  B += c_col * K;

  const int4 offsets_a(0, c_row, base_oh, base_ow);
  const int2 offsets_b(0, c_col);

  // Prepare threadgroup loading operations
  loader_a_t loader_a(
      A,
      As,
      offsets_a,
      params,
      jump_params,
      base_wh,
      base_ww,
      simd_gid,
      simd_lid);
  loader_b_t loader_b(
      B,
      Bs,
      offsets_b,
      params,
      jump_params,
      base_wh,
      base_ww,
      simd_gid,
      simd_lid);

  // Prepare threadgroup mma operation
  mma_t mma_op(simd_gid, simd_lid);

  if (align_C) {
    int gemm_k_iterations =
        base_wh_size * base_ww_size * gemm_params->gemm_k_iterations;

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
  }

  else {
    for (int k = 1; k < gemm_params->gemm_k_iterations; k++) {
      for (int j = 0; j < base_wh_size * base_ww_size; j++) {
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
    }
    const short remaining_k = params->C % BK;
    for (int j = 0; j < base_wh_size * base_ww_size; j++) {
      // Load elements into threadgroup
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(remaining_k);
      loader_b.load_safe(remaining_k);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);
      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results to device memory
  {
    // Adjust for simdgroup and thread location
    int offset_m = c_row + mma_op.sm;
    int offset_n = c_col + mma_op.sn;
    C += offset_n;

    if (offset_n >= gemm_params->N)
      return;

    short diff = gemm_params->N - offset_n;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < mma_t::TM; i++) {
      int cm = offset_m + i * mma_t::TM_stride;

      int n = cm / jump_params->adj_out_hw;
      int hw = cm % jump_params->adj_out_hw;
      int oh =
          (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h + base_oh;
      int ow =
          (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w + base_ow;

      if (n < params->N && oh < params->oS[0] && ow < params->oS[1]) {
        int offset_cm = n * params->out_strides[0] +
            oh * params->out_strides[1] + ow * params->out_strides[2];

        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < mma_t::TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = mma_op.Ctile.frag_at(i, j);
          int offset = offset_cm + (j * mma_t::TN_stride);

          constexpr short kelems = decltype(mma_op.Ctile)::kElemsPerFrag;

          // Apply epilogue and output C
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kelems; k++) {
            if ((j * mma_t::TN_stride + k) < diff) {
              C[offset + k] = Epilogue::apply(accum[k]);
            }
          }
        }
      }
    }
  }
}
