#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)

// Quantized SDPA vector instantiations
// Uses QuantMode enum for explicit mode selection
#define instantiate_quant_sdpa_vector(type, head_dim, mode, group_size, bits) \
  instantiate_kernel(                                                         \
      "quant_sdpa_vector_2pass_1_" #type "_" #head_dim "_" #mode,             \
      quant_sdpa_vector_2pass_1,                                              \
      type,                                                                   \
      head_dim,                                                               \
      QuantMode::mode,                                                        \
      group_size,                                                             \
      bits)

#define instantiate_quant_sdpa_vector_all_modes(type, head_dim) \
  instantiate_quant_sdpa_vector(type, head_dim, Mxfp4, 32, 4)   \
  instantiate_quant_sdpa_vector(type, head_dim, Nvfp4, 16, 4)   \
  instantiate_quant_sdpa_vector(type, head_dim, Mxfp8, 32, 8)

#define instantiate_quant_sdpa_vector_heads(type) \
  instantiate_quant_sdpa_vector_all_modes(type, 64)    \
  instantiate_quant_sdpa_vector_all_modes(type, 128)

instantiate_quant_sdpa_vector_heads(float)
instantiate_quant_sdpa_vector_heads(bfloat16_t)
instantiate_quant_sdpa_vector_heads(float16_t)
    // clang-format on
