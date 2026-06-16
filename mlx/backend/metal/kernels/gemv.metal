// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/gemv.h"

using namespace metal;

#define instantiate_gemv_helper(                                      \
    name, itype, bm, bn, sm, sn, tm, tn, nc, axpby)                   \
  instantiate_kernel(                                                 \
      "gemv_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm \
      "_tn" #tn "_nc" #nc "_axpby" #axpby,                            \
      gemv,                                                           \
      itype,                                                          \
      bm,                                                             \
      bn,                                                             \
      sm,                                                             \
      sn,                                                             \
      tm,                                                             \
      tn,                                                             \
      nc,                                                             \
      axpby)

// clang-format off
#define instantiate_gemv(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 1) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 1) // clang-format on

// clang-format off
#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 4, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 8,  1, 1, 32, 4, 4) // clang-format on

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);
instantiate_gemv_blocks(complex64, complex64_t);

// clang-format off
#define instantiate_gemv_bs_helper(nm, itype, bm, bn, sm, sn, tm, tn) \
  instantiate_kernel(                                                 \
    "gemv_gather_" #nm "_bm" #bm "_bn" #bn "_sm" #sm                  \
                       "_sn" #sn "_tm" #tm "_tn" #tn,                 \
    gemv_gather, itype, bm, bn, sm, sn, tm, tn)

#define instantiate_gemv_bs_blocks(name, itype)              \
  instantiate_gemv_bs_helper(name, itype, 4, 1, 1, 32, 1, 4) \
  instantiate_gemv_bs_helper(name, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_bs_helper(name, itype, 8, 1, 1, 32, 4, 4) // clang-format on

instantiate_gemv_bs_blocks(float32, float);
instantiate_gemv_bs_blocks(float16, half);
instantiate_gemv_bs_blocks(bfloat16, bfloat16_t);
instantiate_gemv_bs_blocks(complex64, complex64_t);

// clang-format off
#define instantiate_gemv_t_helper(                          \
    name, itype, bm, bn, sm, sn, tm, tn, nc, axpby)         \
  instantiate_kernel(                                       \
    "gemv_t_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn \
       "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby,       \
  gemv_t, itype, bm, bn, sm, sn, tm, tn, nc, axpby)

#define instantiate_gemv_t(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 1) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 1) // clang-format on

// clang-format off
#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 1) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 4,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 4, 8, 4, 4) // clang-format on

// clang-format off
instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t);
instantiate_gemv_t_blocks(complex64, complex64_t); // clang-format on

// clang-format off
#define instantiate_gemv_t_bs_helper(                  \
    nm, itype, bm, bn, sm, sn, tm, tn)                 \
  instantiate_kernel(                                  \
    "gemv_t_gather_" #nm "_bm" #bm "_bn" #bn "_sm" #sm \
       "_sn" #sn "_tm" #tm "_tn" #tn,                  \
  gemv_t_gather, itype, bm, bn, sm, sn, tm, tn)

#define instantiate_gemv_t_bs_blocks(name, itype)              \
  instantiate_gemv_t_bs_helper(name, itype, 1,  2, 8, 4, 4, 1) \
  instantiate_gemv_t_bs_helper(name, itype, 1,  2, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1,  4, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1, 16, 4, 8, 4, 4) // clang-format on

// clang-format off
instantiate_gemv_t_bs_blocks(float32, float);
instantiate_gemv_t_bs_blocks(float16, half);
instantiate_gemv_t_bs_blocks(bfloat16, bfloat16_t);
instantiate_gemv_t_bs_blocks(complex64, complex64_t); // clang-format on
