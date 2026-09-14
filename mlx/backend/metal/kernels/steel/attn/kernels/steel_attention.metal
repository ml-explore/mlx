// Copyright © 2024-26 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype, bk256) \
    instantiate_attn(iname, itype, 32, bk256, 256, 4, 2, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 192, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  96, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  72, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype, bk256) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype, bk256) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool, bk256)

instantiate_attn_mask_helper(float16, half, 32);
instantiate_attn_mask_helper(bfloat16, bfloat16_t, 32);
instantiate_attn(float16, half, 32, 16, 256, 4, 2, float16, half);
instantiate_attn(float16, half, 32, 16, 256, 4, 2, bool_, bool);
instantiate_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 2, bfloat16, bfloat16_t);
instantiate_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 2, bool_, bool);

instantiate_attn_mask_helper(float32, float, 16);
// clang-format on
