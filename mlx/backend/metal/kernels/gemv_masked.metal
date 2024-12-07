// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/gemv_masked.h"

#define instantiate_gemv_helper(                                           \
    outm_n, outm_t, opm_n, opm_t, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_kernel(                                                      \
    "gemv_outmask_" #outm_n "_opmask_" #opm_n "_" #name                    \
      "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm                    \
      "_tn" #tn "_nc" #nc,                                                 \
  gemv_masked, itype, outm_t, opm_t, bm, bn, sm, sn, tm, tn, nc)

#define instantiate_gemv_base(name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(bool_, bool, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(name, itype, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(bool_, bool, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(name, itype, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(nomask, nomask_t, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(nomask, nomask_t, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(bool_, bool, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(name, itype, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc)

#define instantiate_gemv(name, itype, bm, bn, sm, sn, tm, tn)   \
  instantiate_gemv_base(name, itype, bm, bn, sm, sn, tm, tn, 0) \
  instantiate_gemv_base(name, itype, bm, bn, sm, sn, tm, tn, 1)

#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 2, 1, 4,  8, 1, 4) \
  instantiate_gemv(name, itype, 2, 1, 4,  8, 4, 4) \
  instantiate_gemv(name, itype, 2, 1, 2, 16, 1, 4) \
  instantiate_gemv(name, itype, 2, 1, 2, 16, 4, 4) \
  instantiate_gemv(name, itype, 4, 1, 2, 16, 4, 4)

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);

#define instantiate_gemv_t_helper(                                           \
    outm_n, outm_t, opm_n, opm_t, name, itype, bm, bn, sm, sn, tm, tn, nc)   \
  instantiate_kernel(                                                        \
    "gemv_t_outmask_" #outm_n "_opmask_" #opm_n "_" #name                    \
      "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm                      \
      "_tn" #tn "_nc" #nc,                                                   \
  gemv_t_masked, itype, outm_t, opm_t, bm, bn, sm, sn, tm, tn, nc)

#define instantiate_gemv_t_base(name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(bool_, bool, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(name, itype, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(bool_, bool, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(name, itype, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(nomask, nomask_t, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(nomask, nomask_t, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(bool_, bool, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(name, itype, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc)

#define instantiate_gemv_t(name, itype, bm, bn, sm, sn, tm, tn)   \
  instantiate_gemv_t_base(name, itype, bm, bn, sm, sn, tm, tn, 0) \
  instantiate_gemv_t_base(name, itype, bm, bn, sm, sn, tm, tn, 1)

#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 1, 1,  8, 4, 4, 1) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 1,  8, 4, 8, 1) \
  instantiate_gemv_t(name, itype, 1, 1,  8, 4, 8, 4) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 8, 4) \
  instantiate_gemv_t(name, itype, 1, 4,  8, 4, 8, 4)

instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t); // clang-format on
