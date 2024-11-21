#include <metal_stdlib>

#include "mlx/backend/metal/kernels/sdpa_vector.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// clang-format off
// SDPA vector instantiations
#define instantiate_sdpa_vector(type, head_dim)                                   \
  instantiate_kernel("sdpa_vector_" #type "_" #head_dim, sdpa_vector, type, head_dim)          \
  instantiate_kernel("sdpa_vector_2pass_1_" #type "_" #head_dim, sdpa_vector_2pass_1, type, head_dim)  \
  instantiate_kernel("sdpa_vector_2pass_2_" #type "_" #head_dim, sdpa_vector_2pass_2, type, head_dim)

#define instantiate_sdpa_vector_heads(type) \
  instantiate_sdpa_vector(type, 64)         \
  instantiate_sdpa_vector(type, 96)         \
  instantiate_sdpa_vector(type, 128)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on
