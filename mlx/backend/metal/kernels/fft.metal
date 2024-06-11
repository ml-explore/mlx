// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/fft.h"

#define instantiate_fft(tg_mem_size, in_T, out_T)   \
  instantiate_kernel(                               \
      "fft_mem_" #tg_mem_size "_" #in_T "_" #out_T, \
      fft,                                          \
      tg_mem_size,                                  \
      in_T,                                         \
      out_T)

#define instantiate_rader(tg_mem_size, in_T, out_T)       \
  instantiate_kernel(                                     \
      "rader_fft_mem_" #tg_mem_size "_" #in_T "_" #out_T, \
      rader_fft,                                          \
      tg_mem_size,                                        \
      in_T,                                               \
      out_T)

#define instantiate_bluestein(tg_mem_size, in_T, out_T)       \
  instantiate_kernel(                                         \
      "bluestein_fft_mem_" #tg_mem_size "_" #in_T "_" #out_T, \
      bluestein_fft,                                          \
      tg_mem_size,                                            \
      in_T,                                                   \
      out_T)

#define instantiate_four_step(tg_mem_size, in_T, out_T, step, real)           \
  instantiate_kernel(                                                         \
      "four_step_mem_" #tg_mem_size "_" #in_T "_" #out_T "_" #step "_" #real, \
      four_step_fft,                                                          \
      tg_mem_size,                                                            \
      in_T,                                                                   \
      out_T,                                                                  \
      step,                                                                   \
      real)

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
