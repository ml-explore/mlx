// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector_turboquant.h"

#define instantiate_sdpa_vector_tq(tname, type, head_dim) \
  template [[host_name("sdpa_vector_turboquant_" #tname "_" #head_dim)]] \
  [[kernel]] void sdpa_vector_turboquant<type, head_dim>( \
      const device type* q_rot [[buffer(0)]], \
      const device type* q_sketch [[buffer(1)]], \
      const device uint8_t* k_packed [[buffer(2)]], \
      const device uint8_t* k_signs [[buffer(3)]], \
      const device float* k_norms [[buffer(4)]], \
      const device float* k_res_norms [[buffer(5)]], \
      const device float* centroids [[buffer(6)]], \
      const device uint8_t* v_packed [[buffer(7)]], \
      const device float* v_scales [[buffer(8)]], \
      const device float* v_zeros [[buffer(9)]], \
      device type* out [[buffer(10)]], \
      const constant mlx::steel::TurboQuantAttnParams& params [[buffer(11)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 tpg [[threadgroups_per_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

// Instantiate for common types and head dimensions
// tname must match get_type_string() output for kernel dispatch to find them
instantiate_sdpa_vector_tq(float16_t, half, 64);
instantiate_sdpa_vector_tq(float16_t, half, 128);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 64);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 128);
instantiate_sdpa_vector_tq(float, float, 64);
instantiate_sdpa_vector_tq(float, float, 128);
// clang-format on
