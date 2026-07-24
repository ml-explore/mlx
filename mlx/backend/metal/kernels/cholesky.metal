// Copyright © 2025 Apple Inc.

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/cholesky.h"

#define instantiate_cholesky(name, itype)                                  \
  instantiate_kernel("cholesky_potf2_" #name, cholesky_potf2, itype)       \
  instantiate_kernel("cholesky_trsm_" #name, cholesky_trsm, itype)         \
  instantiate_kernel("cholesky_syrk32_" #name, cholesky_syrk, itype, 32)   \
  instantiate_kernel("cholesky_syrk64_" #name, cholesky_syrk, itype, 64)   \
  instantiate_kernel("cholesky_fixup_" #name, cholesky_fixup, itype)

instantiate_cholesky(float32, float) // clang-format on
