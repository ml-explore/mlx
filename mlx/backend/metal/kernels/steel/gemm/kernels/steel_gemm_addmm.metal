// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;
using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

template <typename T,
          int BM,
          int BN,
          int BK,
          int WM,
          int WN,
          bool transpose_a, 
          bool transpose_b,
          bool MN_aligned,
          bool K_aligned,
          typename AccumType = float,
          typename Epilogue = TransformAdd<T, AccumType>>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void addmm(
    const device T *A [[buffer(0)]],
    const device T *B [[buffer(1)]],
    const device T *C [[buffer(2)]],
    device T *D [[buffer(3)]],
    const constant GEMMAddMMParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { 
    
    // Pacifying compiler
    (void)lid;
    
    using gemm_kernel = 
        GEMMKernel<T, T, BM, BN, BK, WM, WN, 
        transpose_a, transpose_b, 
        MN_aligned, K_aligned,
        AccumType, Epilogue>;
    
    using loader_a_t = typename gemm_kernel::loader_a_t;
    using loader_b_t = typename gemm_kernel::loader_b_t;
    using mma_t = typename gemm_kernel::mma_t;
    
    threadgroup T As[gemm_kernel::tgp_mem_size_a];
    threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

    // Adjust for batch
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;
    C += params->batch_stride_c * tid.z;
    D += params->batch_stride_d * tid.z;

    const int tid_y = ((tid.y) << params->swizzle_log) +
        ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int tid_x = (tid.x) >> params->swizzle_log;

    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Find block in A, B, C
    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;

    A += transpose_a ? c_row : c_row * params->lda;
    B += transpose_b ? c_col * params->ldb : c_col;
    C += c_row * params->ldc + c_col * params->fdc;
    D += c_row * params->ldd + c_col;

    // Prepare threadgroup loading operations
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    int gemm_k_iterations = params->gemm_k_iterations_aligned;

    const Epilogue epilogue_op(params->alpha, params->beta);

    ///////////////////////////////////////////////////////////////////////////////
    // MNK aligned loop
    if (MN_aligned) {
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

      // Loop tail
      if (!K_aligned) {
        int lbk = params->K - params->gemm_k_iterations_aligned * BK;
        short2 tile_dims_A = transpose_a ? short2(BM, lbk) : short2(lbk, BM);
        short2 tile_dims_B = transpose_b ? short2(lbk, BN) : short2(BN, lbk);

        thread bool mask_A[loader_a_t::n_rows][loader_a_t::vec_size];
        thread bool mask_B[loader_b_t::n_rows][loader_b_t::vec_size];

        loader_a.set_mask(tile_dims_A, mask_A);
        loader_b.set_mask(tile_dims_B, mask_B);

        loader_a.load_safe(mask_A);
        loader_b.load_safe(mask_B);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(As, Bs);
      }

      // Store results to device memory
      mma_op.store_result(D, params->ldd, C, params->ldc, params->fdc, epilogue_op);
      return;

    }
    ///////////////////////////////////////////////////////////////////////////////
    // MN unaligned loop
    else { // Loop over K - unaligned case
      short tgp_bm = min(BM, params->M - c_row);
      short tgp_bn = min(BN, params->N - c_col);
      short leftover_bk = params->K - params->gemm_k_iterations_aligned * BK;

      if (tgp_bm == BM && tgp_bn == BN) {
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
            LoopAlignment<true, true, K_aligned>{});

        mma_op.store_result(D, params->ldd, C, params->ldc, params->fdc, epilogue_op);
        return;

      } else if (tgp_bn == BN) {
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
            LoopAlignment<false, true, K_aligned>{});

        return mma_op.store_result_safe(
            D, params->ldd, 
            C, params->ldc, params->fdc,
            short2(tgp_bn, tgp_bm), 
            epilogue_op);

      } else if (tgp_bm == BM) {
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
            LoopAlignment<true, false, K_aligned>{});

        return mma_op.store_result_safe(
            D, params->ldd, 
            C, params->ldc, params->fdc,
            short2(tgp_bn, tgp_bm), 
            epilogue_op);

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
            LoopAlignment<false, false, K_aligned>{});

        return mma_op.store_result_safe(
            D, params->ldd, 
            C, params->ldc, params->fdc,
            short2(tgp_bn, tgp_bm), 
            epilogue_op);
      }
    }
}

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel initializations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned, ep_name, epilogue) \
  template [[host_name("steel_addmm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_MN_" #aname "_K_" #kname "_" #ep_name)]] \
  [[kernel]] void addmm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, mn_aligned, k_aligned, float, epilogue<itype, float>>( \
      const device itype *A [[buffer(0)]], \
      const device itype *B [[buffer(1)]], \
      const device itype *C [[buffer(2)]], \
      device itype *D [[buffer(3)]], \
      const constant GEMMAddMMParams* params [[buffer(4)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_bias_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned, add, TransformAdd) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned, axpby, TransformAxpby)

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gemm_bias_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true) \
  instantiate_gemm_bias_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm_bias_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm_bias_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 32, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 64, 16, 2, 2)

instantiate_gemm_shapes_helper(float16, half, float16, half);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);

instantiate_gemm_shapes_helper(float32, float, float32, float);