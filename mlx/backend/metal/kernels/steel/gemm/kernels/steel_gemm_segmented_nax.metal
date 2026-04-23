// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_segmented_nax.h"

// clang-format off
#define instantiate_segmented_mm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                          \
      "steel_segmented_mm_nax_" #tname "_" #iname "_" #oname                   \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                        \
  segmented_mm_nax, itype, bm, bn, bk, wm, wn, trans_a, trans_b, float)

#define instantiate_segmented_mm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_segmented_mm(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn)      \
  instantiate_segmented_mm(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn)      \
  instantiate_segmented_mm(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn)      \
  instantiate_segmented_mm(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_segmented_mm_shapes_helper(iname, itype, oname, otype)                 \
  instantiate_segmented_mm_transpose_helper(iname, itype, oname, otype, 64, 64, 256, 2, 2) \
  instantiate_segmented_mm_transpose_helper(iname, itype, oname, otype, 64, 64, 128, 2, 2) \
  instantiate_segmented_mm_transpose_helper(iname, itype, oname, otype, 64, 64, 64, 2, 2)

instantiate_segmented_mm_shapes_helper(float16, half, float16, half);
instantiate_segmented_mm_shapes_helper(bfloat16, bfloat, bfloat16, bfloat);
instantiate_segmented_mm_shapes_helper(float32, float, float32, float);
// clang-format on
