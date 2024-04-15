// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"
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
          bool has_operand_mask=false>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void tile_masked_gemm(
    const device T *A [[buffer(0)]],
    const device T *B [[buffer(1)]],
    device T *D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant size_t* batch_strides [[buffer(7)]],
    const device bool *out_mask [[buffer(10)]],
    const device bool *lhs_mask [[buffer(11)]],
    const device bool *rhs_mask [[buffer(12)]],
    const constant int* mask_strides [[buffer(13)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { 

    // Appease the compiler
    (void)lid;
    
    using gemm_kernel = GEMMKernel<T, T, BM, BN, BK, WM, WN, transpose_a, transpose_b, MN_aligned, K_aligned>;
    
    const int tid_y = ((tid.y) << params->swizzle_log) +
        ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int tid_x = (tid.x) >> params->swizzle_log;

    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    if(params->batch_ndim > 1) {
      const constant size_t* mask_batch_strides = batch_strides + 2 * params->batch_ndim;
      out_mask += elem_to_loc(tid.z, batch_shape, mask_batch_strides, params->batch_ndim);

      if(has_operand_mask) {
        const constant size_t* mask_strides_lhs = mask_batch_strides + params->batch_ndim;
        const constant size_t* mask_strides_rhs = mask_strides_lhs + params->batch_ndim;

        ulong2 batch_offsets = elem_to_loc_broadcast(
            tid.z, batch_shape, mask_strides_lhs, mask_strides_rhs, params->batch_ndim);

        lhs_mask += batch_offsets.x;
        rhs_mask += batch_offsets.y;
      }
    }

    // Adjust for batch
    if(params->batch_ndim > 1) {
      const constant size_t* A_bstrides = batch_strides;
      const constant size_t* B_bstrides = batch_strides + params->batch_ndim;

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

    A += transpose_a ? c_row : c_row * params->lda;
    B += transpose_b ? c_col * params->ldb : c_col;
    D += c_row * params->ldd + c_col;


    bool mask_out = out_mask[tid_y * mask_strides[1] + tid_x * mask_strides[0]];

    // Write zeros and return
    if(!mask_out) {
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
          for(short j = 0; j < vec_size; j++) {
            D[ti * params->ldd + j] = T(0.);
          }
        }
      } else {
        short jmax = tgp_bn - bj;
        jmax = jmax < vec_size ? jmax : vec_size;
        for (short ti = 0; (bi + ti) < tgp_bm; ti += TM) {
          for(short j = 0; j < jmax; j++) {
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
    thread typename gemm_kernel::loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread typename gemm_kernel::loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    ///////////////////////////////////////////////////////////////////////////////
    // MNK aligned loop
    if (MN_aligned) {
      for (int k = 0; k < gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(!has_operand_mask || 
            (lhs_mask[tid_y * mask_strides[3] + ((k * BK) / BM) * mask_strides[2]] && 
             rhs_mask[((k * BK) / BM) * mask_strides[5] + tid_x * mask_strides[4]])) {

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

        if(!has_operand_mask || 
            (lhs_mask[tid_y * mask_strides[3] + (params->K / BM) * mask_strides[2]] && 
             rhs_mask[(params->K / BM) * mask_strides[5] + tid_x * mask_strides[4]])) {

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
        if(!has_operand_mask || 
            (lhs_mask[tid_y * mask_strides[3] + ((k * BK) / BM) * mask_strides[2]] && 
             rhs_mask[((k * BK) / BM) * mask_strides[5] + tid_x * mask_strides[4]])) {

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

        if(!has_operand_mask || 
            (lhs_mask[tid_y * mask_strides[3] + (params->K / BM) * mask_strides[2]] && 
             rhs_mask[(params->K / BM) * mask_strides[5] + tid_x * mask_strides[4]])) {

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

      if(M_aligned && N_aligned) {
        mma_op.store_result(D, params->ldd);
      } else {
        mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
      }
    }
}

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel initializations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned, omname, op_mask) \
  template [[host_name("steel_tile_masked_gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_MN_" #aname "_K_" #kname "_op_mask_" #omname)]] \
  [[kernel]] void tile_masked_gemm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, mn_aligned, k_aligned, op_mask>( \
      const device itype *A [[buffer(0)]], \
      const device itype *B [[buffer(1)]], \
      device itype *D [[buffer(3)]], \
      const constant GEMMParams* params [[buffer(4)]], \
      const constant int* batch_shape [[buffer(6)]], \
      const constant size_t* batch_strides [[buffer(7)]], \
      const device bool *out_mask [[buffer(10)]], \
      const device bool *lhs_mask [[buffer(11)]], \
      const device bool *rhs_mask [[buffer(12)]], \
      const constant int* mask_strides [[buffer(13)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned, N, false) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned, T, true)

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true) \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2)

instantiate_gemm_shapes_helper(float16, half, float16, half);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);
instantiate_gemm_shapes_helper(float32, float, float32, float);