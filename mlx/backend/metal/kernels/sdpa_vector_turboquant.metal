// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector_turboquant.h"

// 1-pass kernel: instantiate for (type, head_dim, mse_bits, v_bits)
#define instantiate_sdpa_vector_tq(tname, type, head_dim, mb, vb) \
  template [[host_name("sdpa_vector_turboquant_" #tname "_" #head_dim "_" #mb "_" #vb)]] \
  [[kernel]] void sdpa_vector_turboquant<type, head_dim, mb, vb>( \
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
      device float* out_m [[buffer(11)]], \
      device float* out_l [[buffer(12)]], \
      const constant mlx::steel::TurboQuantAttnParams& params [[buffer(13)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 tpg [[threadgroups_per_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

// 2-bit instantiations
instantiate_sdpa_vector_tq(float16_t, half, 64, 2, 2);
instantiate_sdpa_vector_tq(float16_t, half, 128, 2, 2);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 64, 2, 2);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 128, 2, 2);
instantiate_sdpa_vector_tq(float, float, 64, 2, 2);
instantiate_sdpa_vector_tq(float, float, 128, 2, 2);

// 4-bit instantiations
instantiate_sdpa_vector_tq(float16_t, half, 64, 4, 4);
instantiate_sdpa_vector_tq(float16_t, half, 128, 4, 4);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 64, 4, 4);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 128, 4, 4);
instantiate_sdpa_vector_tq(float, float, 64, 4, 4);
instantiate_sdpa_vector_tq(float, float, 128, 4, 4);

// Mixed: 4-bit keys, 2-bit values
instantiate_sdpa_vector_tq(float16_t, half, 64, 4, 2);
instantiate_sdpa_vector_tq(float16_t, half, 128, 4, 2);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 64, 4, 2);
instantiate_sdpa_vector_tq(bfloat16_t, bfloat16_t, 128, 4, 2);
instantiate_sdpa_vector_tq(float, float, 64, 4, 2);
instantiate_sdpa_vector_tq(float, float, 128, 4, 2);

// 2-pass kernels: pass 1
#define instantiate_sdpa_vector_tq_2pass_1(tname, type, head_dim, mb, vb) \
  template [[host_name("sdpa_vector_turboquant_2pass_1_" #tname "_" #head_dim "_" #mb "_" #vb)]] \
  [[kernel]] void sdpa_vector_turboquant_2pass_1<type, head_dim, mb, vb>( \
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
      device float* out_sums [[buffer(11)]], \
      device float* out_maxs [[buffer(12)]], \
      const constant mlx::steel::TurboQuantAttnParams& params [[buffer(13)]], \
      uint3 tptg [[threads_per_threadgroup]], \
      uint3 tidtg [[thread_position_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 tpg [[threadgroups_per_grid]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

// 2-bit
instantiate_sdpa_vector_tq_2pass_1(float16_t, half, 64, 2, 2);
instantiate_sdpa_vector_tq_2pass_1(float16_t, half, 128, 2, 2);
instantiate_sdpa_vector_tq_2pass_1(bfloat16_t, bfloat16_t, 64, 2, 2);
instantiate_sdpa_vector_tq_2pass_1(bfloat16_t, bfloat16_t, 128, 2, 2);
instantiate_sdpa_vector_tq_2pass_1(float, float, 64, 2, 2);
instantiate_sdpa_vector_tq_2pass_1(float, float, 128, 2, 2);

// 4-bit
instantiate_sdpa_vector_tq_2pass_1(float16_t, half, 64, 4, 4);
instantiate_sdpa_vector_tq_2pass_1(float16_t, half, 128, 4, 4);
instantiate_sdpa_vector_tq_2pass_1(bfloat16_t, bfloat16_t, 64, 4, 4);
instantiate_sdpa_vector_tq_2pass_1(bfloat16_t, bfloat16_t, 128, 4, 4);
instantiate_sdpa_vector_tq_2pass_1(float, float, 64, 4, 4);
instantiate_sdpa_vector_tq_2pass_1(float, float, 128, 4, 4);

// Mixed: 4-bit keys, 2-bit values
instantiate_sdpa_vector_tq_2pass_1(float16_t, half, 64, 4, 2);
instantiate_sdpa_vector_tq_2pass_1(float16_t, half, 128, 4, 2);
instantiate_sdpa_vector_tq_2pass_1(bfloat16_t, bfloat16_t, 64, 4, 2);
instantiate_sdpa_vector_tq_2pass_1(bfloat16_t, bfloat16_t, 128, 4, 2);
instantiate_sdpa_vector_tq_2pass_1(float, float, 64, 4, 2);
instantiate_sdpa_vector_tq_2pass_1(float, float, 128, 4, 2);

// 2-pass kernels: pass 2 (merge blocks — no bit-width dependency)
#define instantiate_sdpa_vector_tq_2pass_2(tname, type, head_dim) \
  template [[host_name("sdpa_vector_turboquant_2pass_2_" #tname "_" #head_dim)]] \
  [[kernel]] void sdpa_vector_turboquant_2pass_2<type, head_dim>( \
      const device type* partials [[buffer(0)]], \
      const device float* sums [[buffer(1)]], \
      const device float* maxs [[buffer(2)]], \
      device type* out [[buffer(3)]], \
      device float* out_m [[buffer(4)]], \
      device float* out_l [[buffer(5)]], \
      const constant int& blocks [[buffer(6)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_sdpa_vector_tq_2pass_2(float16_t, half, 64);
instantiate_sdpa_vector_tq_2pass_2(float16_t, half, 128);
instantiate_sdpa_vector_tq_2pass_2(bfloat16_t, bfloat16_t, 64);
instantiate_sdpa_vector_tq_2pass_2(bfloat16_t, bfloat16_t, 128);
instantiate_sdpa_vector_tq_2pass_2(float, float, 64);
instantiate_sdpa_vector_tq_2pass_2(float, float, 128);
// clang-format on
