// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/fft.h"

#define instantiate_fft(tg_mem_size, in_T, out_T)        \
  template [[host_name("fft_mem_" #tg_mem_size "_" #in_T \
                       "_" #out_T)]] [[kernel]] void     \
  fft<tg_mem_size, in_T, out_T>(                         \
      const device in_T* in [[buffer(0)]],               \
      device out_T* out [[buffer(1)]],                   \
      constant const int& n,                             \
      constant const int& batch_size,                    \
      uint3 elem [[thread_position_in_grid]],            \
      uint3 grid [[threads_per_grid]]);

#define instantiate_rader(tg_mem_size, in_T, out_T)            \
  template [[host_name("rader_fft_mem_" #tg_mem_size "_" #in_T \
                       "_" #out_T)]] [[kernel]] void           \
  rader_fft<tg_mem_size, in_T, out_T>(                         \
      const device in_T* in [[buffer(0)]],                     \
      device out_T* out [[buffer(1)]],                         \
      const device float2* raders_b_q [[buffer(2)]],           \
      const device short* raders_g_q [[buffer(3)]],            \
      const device short* raders_g_minus_q [[buffer(4)]],      \
      constant const int& n,                                   \
      constant const int& batch_size,                          \
      constant const int& rader_n,                             \
      uint3 elem [[thread_position_in_grid]],                  \
      uint3 grid [[threads_per_grid]]);

#define instantiate_bluestein(tg_mem_size, in_T, out_T)            \
  template [[host_name("bluestein_fft_mem_" #tg_mem_size "_" #in_T \
                       "_" #out_T)]] [[kernel]] void               \
  bluestein_fft<tg_mem_size, in_T, out_T>(                         \
      const device in_T* in [[buffer(0)]],                         \
      device out_T* out [[buffer(1)]],                             \
      const device float2* w_q [[buffer(2)]],                      \
      const device float2* w_k [[buffer(3)]],                      \
      constant const int& length,                                  \
      constant const int& n,                                       \
      constant const int& batch_size,                              \
      uint3 elem [[thread_position_in_grid]],                      \
      uint3 grid [[threads_per_grid]]);

#define instantiate_four_step(tg_mem_size, in_T, out_T, step, real)       \
  template [[host_name("four_step_mem_" #tg_mem_size "_" #in_T "_" #out_T \
                       "_" #step "_" #real)]] [[kernel]] void             \
  four_step_fft<tg_mem_size, in_T, out_T, step, real>(                    \
      const device in_T* in [[buffer(0)]],                                \
      device out_T* out [[buffer(1)]],                                    \
      constant const int& n1,                                             \
      constant const int& n2,                                             \
      constant const int& batch_size,                                     \
      uint3 elem [[thread_position_in_grid]],                             \
      uint3 grid [[threads_per_grid]]);

// clang-format off
#define instantiate_ffts(tg_mem_size)                        \
  instantiate_fft(tg_mem_size, float2, float2) \
  instantiate_fft(tg_mem_size, float, float2) \
  instantiate_fft(tg_mem_size, float2, float) \
  instantiate_rader(tg_mem_size, float2, float2) \
  instantiate_rader(tg_mem_size, float, float2) \
  instantiate_rader(tg_mem_size, float2, float) \
  instantiate_bluestein(tg_mem_size, float2, float2) \
  instantiate_bluestein(tg_mem_size, float, float2) \
  instantiate_bluestein(tg_mem_size, float2, float) \
  instantiate_four_step(tg_mem_size, float2, float2, 0, /*real=*/false) \
  instantiate_four_step(tg_mem_size, float2, float2, 1, /*real=*/false) \
  instantiate_four_step(tg_mem_size, float, float2, 0, /*real=*/true) \
  instantiate_four_step(tg_mem_size, float2, float2, 1, /*real=*/true) \
  instantiate_four_step(tg_mem_size, float2, float2, 0, /*real=*/true) \
  instantiate_four_step(tg_mem_size, float2, float, 1, /*real=*/true)

// It's substantially faster to statically define the
// threadgroup memory size rather than using
// `setThreadgroupMemoryLength` on the compute encoder.
// For non-power of 2 sizes we round up the shared memory.
instantiate_ffts(256)
instantiate_ffts(512)
instantiate_ffts(1024)
instantiate_ffts(2048)
// 4096 is the max that will fit into 32KB of threadgroup memory.
instantiate_ffts(4096) // clang-format on
