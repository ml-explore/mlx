// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/conv/conv.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.h"

#define instantiate_implicit_conv_3d(                     \
    name,                                                 \
    itype,                                                \
    bm,                                                   \
    bn,                                                   \
    bk,                                                   \
    wm,                                                   \
    wn,                                                   \
    fn,                                                   \
    f)                                                    \
  instantiate_kernel(                                     \
      "implicit_gemm_conv_3d_" #name "_bm" #bm "_bn" #bn  \
          "_bk" #bk "_wm" #wm "_wn" #wn "_filter_" #fn,   \
      implicit_gemm_conv_3d,                              \
      itype,                                              \
      bm,                                                 \
      bn,                                                 \
      bk,                                                 \
      wm,                                                 \
      wn,                                                 \
      f)

#define instantiate_implicit_conv_3d_filter(name, itype, bm, bn, bk, wm, wn)  \
    instantiate_implicit_conv_3d(name, itype, bm, bn, bk, wm, wn, s, true)    \
    instantiate_implicit_conv_3d(name, itype, bm, bn, bk, wm, wn, l, false)

#define instantiate_implicit_3d_blocks(name, itype)                       \
    instantiate_implicit_conv_3d_filter(name, itype, 32,  8, 16, 4, 1)    \
    instantiate_implicit_conv_3d_filter(name, itype, 64,  8, 16, 4, 1)    \
    instantiate_implicit_conv_3d_filter(name, itype, 32, 32, 16, 2, 2)    \
    instantiate_implicit_conv_3d_filter(name, itype, 32, 64, 16, 2, 2)    \
    instantiate_implicit_conv_3d_filter(name, itype, 64, 32, 16, 2, 2)    \
    instantiate_implicit_conv_3d_filter(name, itype, 64, 64, 16, 2, 2)

instantiate_implicit_3d_blocks(float32, float);
instantiate_implicit_3d_blocks(float16, half);
instantiate_implicit_3d_blocks(bfloat16, bfloat16_t); // clang-format on
