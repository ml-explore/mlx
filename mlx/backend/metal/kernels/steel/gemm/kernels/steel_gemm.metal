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
          bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm(
    const device T *A [[buffer(0)]],
    const device T *B [[buffer(1)]],
    device T *C [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { 
    
    using gemm_kernel = GEMMKernel<T, T, BM, BN, BK, WM, WN, transpose_a, transpose_b, MN_aligned, K_aligned>;
    
    threadgroup T As[gemm_kernel::tgp_mem_size_a];
    threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

    // Adjust for batch
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;
    C += params->batch_stride_c * tid.z;

    gemm_kernel::run( 
      A, B, C, 
      params,
      As, Bs,
      simd_lane_id, simd_group_id, tid, lid
    );
}

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel initializations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned) \
  template [[host_name("steel_gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_MN_" #aname "_K_" #kname)]] \
  [[kernel]] void gemm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, mn_aligned, k_aligned>( \
      const device itype *A [[buffer(0)]], \
      const device itype *B [[buffer(1)]], \
      device itype *C [[buffer(2)]], \
      const constant GEMMParams* params [[buffer(3)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

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