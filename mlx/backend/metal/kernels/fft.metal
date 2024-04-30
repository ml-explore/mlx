// Copyright © 2024 Apple Inc.

// Metal FFT using Stockham's algorithm
//
// References:
// - VkFFT (https://github.com/DTolm/VkFFT)
// - Eric Bainville's excellent page (http://www.bealto.com/gpu-fft.html)

#include <metal_common>
#include <metal_math>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

float2 complex_mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

float2 get_twiddle(int k, int p) {
  float theta = -1.0f * k * M_PI_F / (2 * p);

  float2 twiddle;
  twiddle.x = metal::fast::cos(theta);
  twiddle.y = metal::fast::sin(theta);
  return twiddle;
}

// single threaded radix2 implemetation
void radix2(
    int i,
    int p,
    int m,
    threadgroup float2* read_buf,
    threadgroup float2* write_buf) {
  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];

  // The index within this sub-DFT
  int k = i & (p - 1);

  float2 twiddle = get_twiddle(k, p);

  float2 z = complex_mul(x_1, twiddle);

  float2 y_0 = x_0 + z;
  float2 y_1 = x_0 - z;

  int j = (i << 1) - k;

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
}

// single threaded radix4 implemetation
void radix4(
    int i,
    int p,
    int m,
    threadgroup float2* read_buf,
    threadgroup float2* write_buf) {
  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2 * m];
  float2 x_3 = read_buf[i + 3 * m];

  // The index within this sub-DFT
  int k = i & (p - 1);

  float2 twiddle = get_twiddle(k, p);
  // e^a * e^b = e^(a + b)
  float2 twiddle_2 = complex_mul(twiddle, twiddle);
  float2 twiddle_3 = complex_mul(twiddle, twiddle_2);

  x_1 = complex_mul(x_1, twiddle);
  x_2 = complex_mul(x_2, twiddle_2);
  x_3 = complex_mul(x_3, twiddle_3);

  float2 minus_i;
  minus_i.x = 0;
  minus_i.y = -1;

  // Hard coded twiddle factors for DFT4
  float2 z_0 = x_0 + x_2;
  float2 z_1 = x_0 - x_2;
  float2 z_2 = x_1 + x_3;
  float2 z_3 = complex_mul(x_1 - x_3, minus_i);

  float2 y_0 = z_0 + z_2;
  float2 y_1 = z_1 + z_3;
  float2 y_2 = z_0 - z_2;
  float2 y_3 = z_1 - z_3;

  int j = ((i - k) << 2) + k;

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
  write_buf[j + 2 * p] = y_2;
  write_buf[j + 3 * p] = y_3;
}

// Each FFT is computed entirely in shared GPU memory.
//
// N is decomposed into radix-2 and radix-4 DFTs:
// e.g. 128 = 2 * 4 * 4 * 4
//
// At each step we use n / 4 threads, each performing
// a single-threaded radix-4 or radix-2 DFT.
//
// We provide the number of radix-2 and radix-4
// steps at compile time for a ~20% performance boost.
template <size_t n, size_t radix_2_steps, size_t radix_4_steps>
[[kernel]] void fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
  // Index of the DFT in batch
  int batch_idx = thread_position_in_grid.x * n;
  // The index in the DFT we're working on
  int i = thread_position_in_grid.y;
  // The number of the threads we're using for each DFT
  int m = threads_per_grid.y;

  // Allocate 2 shared memory buffers for Stockham.
  // We alternate reading from one and writing to the other at each radix step.
  threadgroup float2 shared_in[n];
  threadgroup float2 shared_out[n];

  // Pointers to facilitate Stockham buffer swapping
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;
  threadgroup float2* tmp;

  // Copy input into shared memory
  shared_in[i] = in[batch_idx + i];
  shared_in[i + m] = in[batch_idx + i + m];
  shared_in[i + 2 * m] = in[batch_idx + i + 2 * m];
  shared_in[i + 3 * m] = in[batch_idx + i + 3 * m];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;

  for (size_t r = 0; r < radix_2_steps; r++) {
    radix2(i, p, m * 2, read_buf, write_buf);
    radix2(i + m, p, m * 2, read_buf, write_buf);
    p *= 2;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stockham switch of buffers
    tmp = write_buf;
    write_buf = read_buf;
    read_buf = tmp;
  }

  for (size_t r = 0; r < radix_4_steps; r++) {
    radix4(i, p, m, read_buf, write_buf);
    p *= 4;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stockham switch of buffers
    tmp = write_buf;
    write_buf = read_buf;
    read_buf = tmp;
  }

  // Copy shared memory to output
  out[batch_idx + i] = read_buf[i];
  out[batch_idx + i + m] = read_buf[i + m];
  out[batch_idx + i + 2 * m] = read_buf[i + 2 * m];
  out[batch_idx + i + 3 * m] = read_buf[i + 3 * m];
}

#define instantiate_fft(name, n, radix_2_steps, radix_4_steps)   \
  template [[host_name("fft_" #name)]] [[kernel]] void           \
  fft<n, radix_2_steps, radix_4_steps>(                          \
      const device float2* in [[buffer(0)]],                     \
      device float2* out [[buffer(1)]],                          \
      uint3 thread_position_in_grid [[thread_position_in_grid]], \
      uint3 threads_per_grid [[threads_per_grid]]);

// Explicitly define kernels for each power of 2.
// clang-format off
instantiate_fft(4, /* n= */ 4, /* radix_2_steps= */ 0, /* radix_4_steps= */ 1)
instantiate_fft(8, 8, 1, 1) instantiate_fft(16, 16, 0, 2)
instantiate_fft(32, 32, 1, 2) instantiate_fft(64, 64, 0, 3)
instantiate_fft(128, 128, 1, 3) instantiate_fft(256, 256, 0, 4)
instantiate_fft(512, 512, 1, 4)
instantiate_fft(1024, 1024, 0, 5)
// 2048 is the max that will fit into 32KB of threadgroup memory.
// TODO: implement 4 step FFT for larger n.
instantiate_fft(2048, 2048, 1, 5) // clang-format on
