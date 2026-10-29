// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_thin_nax.h"

// clang-format off
#define instantiate_gemm_thin(tname, trans_b, iname, itype, um, un, sm, sn, ks) \
  instantiate_kernel(                                                         \
      "steel_gemm_thin_nax_" #tname "_" #iname                                \
      "_um" #um "_un" #un "_sm" #sm "_sn" #sn "_ks" #ks,                      \
  gemm_thin_nax, itype, um, un, sm, sn, ks, trans_b)

#define instantiate_gemm_thin_transpose_helper(iname, itype, um, un, sm, sn, ks) \
    instantiate_gemm_thin(nn, false, iname, itype, um, un, sm, sn, ks) \
    instantiate_gemm_thin(nt, true , iname, itype, um, un, sm, sn, ks)

#define instantiate_gemm_thin_shapes_helper(iname, itype) \
    instantiate_gemm_thin_transpose_helper(iname, itype, 2, 2, 1, 1, 4) \
    instantiate_gemm_thin_transpose_helper(iname, itype, 2, 2, 1, 2, 2)

instantiate_gemm_thin_shapes_helper(float16, half);
instantiate_gemm_thin_shapes_helper(bfloat16, bfloat);
// clang-format on
