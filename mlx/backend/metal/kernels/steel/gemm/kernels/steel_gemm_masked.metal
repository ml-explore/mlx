// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_masked.h"

#define instantiate_gemm(                                              \
    outmaskname,                                                       \
    outmasktype,                                                       \
    opmaskname,                                                        \
    opmasktype,                                                        \
    tname,                                                             \
    trans_a,                                                           \
    trans_b,                                                           \
    iname,                                                             \
    itype,                                                             \
    oname,                                                             \
    otype,                                                             \
    bm,                                                                \
    bn,                                                                \
    bk,                                                                \
    wm,                                                                \
    wn,                                                                \
    aname,                                                             \
    mn_aligned,                                                        \
    kname,                                                             \
    k_aligned)                                                         \
  instantiate_kernel(                                                  \
    "steel_gemm_block_outmask_" #outmaskname                           \
      "_opmask_" #opmaskname "_" #tname "_" #iname "_" #oname          \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn                \
      "_MN_" #aname "_K_" #kname,                                      \
    block_masked_gemm,                                                 \
      itype,                                                           \
      outmasktype,                                                     \
      opmasktype,                                                      \
      bm,                                                              \
      bn,                                                              \
      bk,                                                              \
      wm,                                                              \
      wn,                                                              \
      trans_a,                                                         \
      trans_b,                                                         \
      mn_aligned,                                                      \
      k_aligned)

#define instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)                \
  instantiate_gemm(bool_, bool, bool_, bool, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)        \
  instantiate_gemm(iname, itype, iname, itype, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)      \
  instantiate_gemm(bool_, bool, iname, itype, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)       \
  instantiate_gemm(iname, itype, bool_, bool, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)       \
  instantiate_gemm(nomask, nomask_t, bool_, bool, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)   \
  instantiate_gemm(nomask, nomask_t, iname, itype, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)  \
  instantiate_gemm(bool_, bool, nomask, nomask_t, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)   \
  instantiate_gemm(iname, itype, nomask, nomask_t, tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned)

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn)                         \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true)  \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm_mask_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn)             \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype)                  \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2)

instantiate_gemm_shapes_helper(float16, half, float16, half);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);
instantiate_gemm_shapes_helper(float32, float, float32, float); // clang-format on
