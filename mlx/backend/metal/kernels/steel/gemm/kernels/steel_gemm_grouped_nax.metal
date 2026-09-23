// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_grouped_nax.h"

// clang-format off
#define instantiate_grouped_mm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                           \
      "steel_grouped_mm_nax_" #tname "_" #iname "_" #oname "_bm" #bm "_bn" #bn  \
      "_bk" #bk "_wm" #wm "_wn" #wn,                                            \
      grouped_mm_nax,                                                           \
      itype,                                                                    \
      bm,                                                                       \
      bn,                                                                       \
      bk,                                                                       \
      wm,                                                                       \
      wn,                                                                       \
      trans_a,                                                                  \
      trans_b,                                                                  \
      float)

#define instantiate_grouped_mm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_grouped_mm(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn)      \
  instantiate_grouped_mm(nt, false,  true, iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_grouped_mm_shapes_helper(iname, itype, oname, otype)                   \
  instantiate_grouped_mm_transpose_helper(iname, itype, oname, otype, 16, 128, 128, 1, 4)  \
  instantiate_grouped_mm_transpose_helper(iname, itype, oname, otype, 32, 128, 128, 1, 4)  \
  instantiate_grouped_mm_transpose_helper(iname, itype, oname, otype, 64, 128, 128, 2, 4)
// clang-format on

instantiate_grouped_mm_shapes_helper(float16, half, float16, half);
instantiate_grouped_mm_shapes_helper(bfloat16, bfloat, bfloat16, bfloat);
instantiate_grouped_mm_shapes_helper(float32, float, float32, float);
