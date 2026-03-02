// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h"

// clang-format off
#define instantiate_gemm_splitk(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                                             \
      "steel_gemm_splitk_nax_" #tname "_"  #iname "_" #oname                                      \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                                          \
  gemm_splitk_nax, itype, bm, bn, bk, wm, wn, trans_a, trans_b, float)

#define instantiate_gemm_splitk_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_splitk(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_splitk(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_splitk(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_splitk(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_splitk_shapes_helper(iname, itype, oname, otype) \
    instantiate_gemm_splitk_transpose_helper(iname, itype, oname, otype,  64,  64, 256, 2, 2) \
    instantiate_gemm_splitk_transpose_helper(iname, itype, oname, otype, 128, 128, 512, 4, 4)

instantiate_gemm_splitk_shapes_helper(float16, half, float32, float);
instantiate_gemm_splitk_shapes_helper(bfloat16, bfloat, float32, float);
instantiate_gemm_splitk_shapes_helper(float32, float, float32, float);
// clang-format on
