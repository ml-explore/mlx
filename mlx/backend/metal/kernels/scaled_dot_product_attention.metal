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

// Quantized SDPA vector instantiations
#define instantiate_quant_sdpa_vector(type, head_dim, group_size, bits) \
  instantiate_kernel(                                                   \
    "quant_sdpa_vector_2pass_1_" #type "_" #head_dim "_" #group_size "_" #bits, \
    quant_sdpa_vector_2pass_1, type, head_dim, group_size, bits)

#define instantiate_quant_sdpa_vector_bits(type, heads, group_size) \
  instantiate_quant_sdpa_vector(type, heads, group_size, 4)         \
  instantiate_quant_sdpa_vector(type, heads, group_size, 8)

#define instantiate_quant_sdpa_vector_group_size(type, heads) \
  instantiate_quant_sdpa_vector_bits(type, heads, 32)         \
  instantiate_quant_sdpa_vector_bits(type, heads, 64)         \
  instantiate_quant_sdpa_vector_bits(type, heads, 128)

#define instantiate_quant_sdpa_vector_heads(type) \
  instantiate_quant_sdpa_vector_group_size(type, 64)         \
  instantiate_quant_sdpa_vector_group_size(type, 128)

instantiate_quant_sdpa_vector_heads(float)
instantiate_quant_sdpa_vector_heads(bfloat16_t)
instantiate_quant_sdpa_vector_heads(float16_t)

    // clang-format on
