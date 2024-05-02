// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;
using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

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
    bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void bs_gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant size_t* batch_strides [[buffer(7)]],
    const constant uint32_t* lhs_indices [[buffer(10)]],
    const constant uint32_t* rhs_indices [[buffer(11)]],
    const constant int* batch_shape_A [[buffer(12)]],
    const constant size_t* batch_strides_A [[buffer(13)]],
    const constant int* batch_shape_B [[buffer(14)]],
    const constant size_t* batch_strides_B [[buffer(15)]],
    const constant int2& operand_batch_ndim [[buffer(16)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
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

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  uint32_t indx_A;
  uint32_t indx_B;

  // Adjust for batch
  if (params->batch_ndim > 1) {
    const constant size_t* A_bstrides = batch_strides;
    const constant size_t* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    indx_A = lhs_indices[batch_offsets.x];
    indx_B = rhs_indices[batch_offsets.y];

  } else {
    indx_A = lhs_indices[params->batch_stride_a * tid.z];
    indx_B = rhs_indices[params->batch_stride_b * tid.z];
  }

  int batch_ndim_A = operand_batch_ndim.x;
  int batch_ndim_B = operand_batch_ndim.y;

  if (batch_ndim_A > 1) {
    A += elem_to_loc(indx_A, batch_shape_A, batch_strides_A, batch_ndim_A);
  } else {
    A += indx_A * batch_strides_A[0];
  }

  if (batch_ndim_B > 1) {
    B += elem_to_loc(indx_B, batch_shape_B, batch_strides_B, batch_ndim_B);
  } else {
    B += indx_B * batch_strides_B[0];
  }

  D += params->batch_stride_d * tid.z;

  gemm_kernel::run(
      A, B, D, params, As, Bs, simd_lane_id, simd_group_id, tid, lid);
}

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel initializations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemm(                                                      \
    tname,                                                                     \
    trans_a,                                                                   \
    trans_b,                                                                   \
    iname,                                                                     \
    itype,                                                                     \
    oname,                                                                     \
    otype,                                                                     \
    bm,                                                                        \
    bn,                                                                        \
    bk,                                                                        \
    wm,                                                                        \
    wn,                                                                        \
    aname,                                                                     \
    mn_aligned,                                                                \
    kname,                                                                     \
    k_aligned)                                                                 \
  template [[host_name("steel_block_sparse_gemm_" #tname "_" #iname "_" #oname \
                       "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn       \
                       "_MN_" #aname "_K_" #kname)]] [[kernel]] void           \
  bs_gemm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, mn_aligned, k_aligned>( \
      const device itype* A [[buffer(0)]],                                     \
      const device itype* B [[buffer(1)]],                                     \
      device itype* D [[buffer(3)]],                                           \
      const constant GEMMParams* params [[buffer(4)]],                         \
      const constant int* batch_shape [[buffer(6)]],                           \
      const constant size_t* batch_strides [[buffer(7)]],                      \
      const constant uint32_t* lhs_indices [[buffer(10)]],                     \
      const constant uint32_t* rhs_indices [[buffer(11)]],                     \
      const constant int* batch_shape_A [[buffer(12)]],                        \
      const constant size_t* batch_strides_A [[buffer(13)]],                   \
      const constant int* batch_shape_B [[buffer(14)]],                        \
      const constant size_t* batch_strides_B [[buffer(15)]],                   \
      const constant int2& operand_batch_ndim [[buffer(16)]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                   \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint3 lid [[thread_position_in_threadgroup]]);

// clang-format off
#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn)             \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true)  \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false) // clang-format on

// clang-format off
#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn)             \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn) // clang-format on

// clang-format off
#define instantiate_gemm_shapes_helper(iname, itype, oname, otype)                  \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 32, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 64, 16, 2, 2) // clang-format on

// clang-format off
instantiate_gemm_shapes_helper(float16, half, float16, half);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);

instantiate_gemm_shapes_helper(float32, float, float32, float); // clang-format on