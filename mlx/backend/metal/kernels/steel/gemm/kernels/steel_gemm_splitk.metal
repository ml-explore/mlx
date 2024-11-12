// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"

#define instantiate_gemm(                                                \
    tname,                                                               \
    trans_a,                                                             \
    trans_b,                                                             \
    iname,                                                               \
    itype,                                                               \
    oname,                                                               \
    otype,                                                               \
    bm,                                                                  \
    bn,                                                                  \
    bk,                                                                  \
    wm,                                                                  \
    wn,                                                                  \
    aname,                                                               \
    mn_aligned,                                                          \
    kname,                                                               \
    k_aligned)                                                           \
  template [[host_name("steel_gemm_splitk_" #tname "_" #iname "_" #oname \
                       "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn \
                       "_MN_" #aname "_K_" #kname)]] [[kernel]] void     \
  gemm_splitk<                                                           \
      itype,                                                             \
      otype,                                                             \
      bm,                                                                \
      bn,                                                                \
      bk,                                                                \
      wm,                                                                \
      wn,                                                                \
      trans_a,                                                           \
      trans_b,                                                           \
      mn_aligned,                                                        \
      k_aligned>(                                                        \
      const device itype* A [[buffer(0)]],                               \
      const device itype* B [[buffer(1)]],                               \
      device otype* C [[buffer(2)]],                                     \
      const constant GEMMSpiltKParams* params [[buffer(3)]],             \
      uint simd_lane_id [[thread_index_in_simdgroup]],                   \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],             \
      uint3 tid [[threadgroup_position_in_grid]],                        \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn)             \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true)  \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn)             \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype)                  \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 16, 16, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 16, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 16, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2)

instantiate_gemm_shapes_helper(float16, half, float32, float);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, float32, float);
instantiate_gemm_shapes_helper(float32, float, float32, float);

#define instantiate_accum(oname, otype, aname, atype)               \
  template [[host_name("steel_gemm_splitk_accum_" #oname            \
                       "_" #aname)]] [[kernel]] void                \
  gemm_splitk_accum<atype, otype>(                                  \
      const device atype* C_split [[buffer(0)]],                    \
      device otype* D [[buffer(1)]],                                \
      const constant int& k_partitions [[buffer(2)]],               \
      const constant int& partition_stride [[buffer(3)]],           \
      const constant int& ldd [[buffer(4)]],                        \
      uint2 gid [[thread_position_in_grid]]);                       \
  template [[host_name("steel_gemm_splitk_accum_" #oname "_" #aname \
                       "_axbpy")]] [[kernel]] void                  \
  gemm_splitk_accum_axpby<atype, otype>(                            \
      const device atype* C_split [[buffer(0)]],                    \
      device otype* D [[buffer(1)]],                                \
      const constant int& k_partitions [[buffer(2)]],               \
      const constant int& partition_stride [[buffer(3)]],           \
      const constant int& ldd [[buffer(4)]],                        \
      const device otype* C [[buffer(5)]],                          \
      const constant int& ldc [[buffer(6)]],                        \
      const constant int& fdc [[buffer(7)]],                        \
      const constant float& alpha [[buffer(8)]],                    \
      const constant float& beta [[buffer(9)]],                     \
      uint2 gid [[thread_position_in_grid]]);

instantiate_accum(bfloat16, bfloat16_t, float32, float);
instantiate_accum(float16, half, float32, float);
instantiate_accum(float32, float, float32, float); // clang-format on
