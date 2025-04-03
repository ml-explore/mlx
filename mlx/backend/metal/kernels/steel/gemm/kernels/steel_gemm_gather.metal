// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.h"

#define instantiate_gather_mm_rhs(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                         \
      "steel_gather_mm_rhs_" #tname "_" #iname "_" #oname "_bm" #bm "_bn" #bn \
      "_bk" #bk "_wm" #wm "_wn" #wn,                                          \
      gather_mm_rhs,                                                          \
      itype,                                                                  \
      bm,                                                                     \
      bn,                                                                     \
      bk,                                                                     \
      wm,                                                                     \
      wn,                                                                     \
      trans_a,                                                                \
      trans_b,                                                                \
      float)

#define instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gather_mm_rhs(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn)  \
  instantiate_gather_mm_rhs(nt, false,  true, iname, itype, oname, otype, bm, bn, bk, wm, wn)  \
  instantiate_gather_mm_rhs(tn,  true, false, iname, itype, oname, otype, bm, bn, bk, wm, wn)  \
  instantiate_gather_mm_rhs(tt,  true,  true, iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gather_mm_shapes_helper(iname, itype, oname, otype)                \
  instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, 16, 64, 16, 1, 2)
// clang-format on

instantiate_gather_mm_shapes_helper(float16, half, float16, half);
instantiate_gather_mm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);
instantiate_gather_mm_shapes_helper(float32, float, float32, float);
