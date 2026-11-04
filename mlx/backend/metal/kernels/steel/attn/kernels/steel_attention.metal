// Copyright © 2024-26 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, bv, wm, wn, mname, mtype) \
  instantiate_kernel(                                                        \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd "_bv" #bv      \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                    \
  attention, dtype, bq, bk, bd, bv, wm, wn, mtype, float)

#define instantiate_attn_helper(iname, itype, bq, bk, bd, bv, wm, wn) \
    instantiate_attn(iname, itype, bq, bk, bd, bv, wm, wn, iname, itype) \
    instantiate_attn(iname, itype, bq, bk, bd, bv, wm, wn, bool_, bool)

#define instantiate_attn_shapes_helper(iname, itype)  \
    instantiate_attn_helper(iname, itype, 32, 16, 192, 192, 4, 1) \
    instantiate_attn_helper(iname, itype, 32, 16, 128, 128, 4, 1) \
    instantiate_attn_helper(iname, itype, 32, 32,  96,  96, 4, 1) \
    instantiate_attn_helper(iname, itype, 32, 32,  96,  64, 4, 1) \
    instantiate_attn_helper(iname, itype, 32, 32,  80,  80, 4, 1) \
    instantiate_attn_helper(iname, itype, 32, 32,  72,  72, 4, 1) \
    instantiate_attn_helper(iname, itype, 32, 32,  64,  64, 4, 1)

instantiate_attn_shapes_helper(float16, half);
instantiate_attn_shapes_helper(bfloat16, bfloat16_t);
instantiate_attn_shapes_helper(float32, float);

instantiate_attn_helper(float16, half, 32, 16, 256, 256, 4, 2);
instantiate_attn_helper(float16, half, 32, 32, 256, 256, 4, 2);
instantiate_attn_helper(bfloat16, bfloat16_t, 32, 16, 256, 256, 4, 2);
instantiate_attn_helper(bfloat16, bfloat16_t, 32, 32, 256, 256, 4, 2);
instantiate_attn_helper(float32, float, 32, 16, 256, 256, 4, 2);

#define instantiate_attn_split(tname, dtype, bq, bk, bd, wm, wn)            \
  instantiate_kernel(                                                        \
      "steel_attention_split_" #tname "_bq" #bq "_bk" #bk "_bd" #bd "_wm" #wm \
      "_wn" #wn,                                                             \
  attention_split, dtype, bq, bk, bd, wm, wn, float)

#define instantiate_attn_split_heads(tname, dtype)          \
  instantiate_attn_split(tname, dtype, 32, 32, 64, 4, 1);  \
  instantiate_attn_split(tname, dtype, 32, 32, 72, 4, 1);  \
  instantiate_attn_split(tname, dtype, 32, 32, 80, 4, 1);  \
  instantiate_attn_split(tname, dtype, 32, 32, 96, 4, 1);  \
  instantiate_attn_split(tname, dtype, 32, 32, 128, 4, 2); \
  instantiate_attn_split(tname, dtype, 32, 32, 192, 4, 2); \
  instantiate_attn_split(tname, dtype, 32, 16, 256, 4, 2);

#define instantiate_attn_split_types(tname, dtype)          \
  instantiate_attn_split_heads(tname, dtype)               \
  instantiate_attn_split(tname, dtype, 32, 32, 256, 4, 2); \
  instantiate_attn_split(tname, dtype, 48, 32, 256, 6, 2); \
  instantiate_attn_split(tname, dtype, 48, 32, 256, 6, 1);

instantiate_attn_split_types(float16, half)
instantiate_attn_split_types(bfloat16, bfloat16_t)
instantiate_attn_split_heads(float32, float)
// clang-format on
