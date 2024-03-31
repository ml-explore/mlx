// Copyright © 2024 Apple Inc.

// Metal FFT using Stockham's algorithm
//
// References:
// - VkFFT (https://github.com/DTolm/VkFFT)
// - Eric Bainville's excellent page (http://www.bealto.com/gpu-fft.html)

#include <metal_math>
#include <metal_common>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define MAX_RADIX_SIZE 4

float2 complex_mul(float2 a, float2 b) {
  float2 c = {
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  };
  return c;
}

float2 get_twiddle(int k, int p) {
  float theta = -1.0f * k * M_PI_F / p;

  float2 twiddle = {
      metal::fast::cos(theta),
      metal::fast::sin(theta)
  };
  return twiddle;
}

// single threaded radix2 implemetation
void radix2(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  // i: the index in the overall DFT that we're processing.
  // p: the size of the DFTs we're merging at this step.
  // m: how many threads are working on this DFT.

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];

  // The index within this sub-DFT
  int k = i & (p - 1);

  float2 twiddle = get_twiddle(k, 2*p);

  float2 z = complex_mul(x_1, twiddle);

  float2 y_0 = x_0 + z;
  float2 y_1 = x_0 - z;

  int j = (i << 1) - k;

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
}

// single threaded radix4 implemetation
void radix4(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2*m];
  float2 x_3 = read_buf[i + 3*m];

  // The index within this sub-DFT
  int k = i & (p - 1);

  float2 twiddle = get_twiddle(k, 2*p);
  // e^a * e^b = e^(a + b)
  float2 twiddle_2 = complex_mul(twiddle, twiddle);
  float2 twiddle_3 = complex_mul(twiddle, twiddle_2);

  x_1 = complex_mul(x_1, twiddle);
  x_2 = complex_mul(x_2, twiddle_2);
  x_3 = complex_mul(x_3, twiddle_3);

  float2 minus_i = {0, -1};

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
  write_buf[j + 2*p] = y_2;
  write_buf[j + 3*p] = y_3;
}

void perform_fft(
    int i,
    int m,
    int radix_2_steps,
    int radix_4_steps,
    threadgroup float2** read_buf,
    threadgroup float2** write_buf) {

  threadgroup float2* tmp;

  int p = 1;

  for (int r = 0; r < radix_2_steps; r++) {
    radix2(i, p, m*2, *read_buf, *write_buf);
    radix2(i + m, p, m*2, *read_buf, *write_buf);
    p *= 2;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stockham switch of buffers
    tmp = *write_buf;
    *write_buf = *read_buf;
    *read_buf = tmp;
  }

  for (int r = 0; r < radix_4_steps; r++) {
    radix4(i, p, m, *read_buf, *write_buf);
    p *= 4;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stockham switch of buffers
    tmp = *write_buf;
    *write_buf = *read_buf;
    *read_buf = tmp;
  }
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
template <size_t n, bool inv, size_t radix_2_steps, size_t radix_4_steps>
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

  // Copy input into shared memory
  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    shared_in[i + t * m] = in[batch_idx + i + t * m];
  }

  // ifft(x) = (1/n)conj(fft(conj(x)))
  if (inv) {
    for (int t = 0; t < MAX_RADIX_SIZE; t++) {
      shared_in[i + t * m].y = -shared_in[i + t * m].y;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, radix_2_steps, radix_4_steps, &read_buf, &write_buf);

  if (inv) {
    float2 inv_factor = {1.0f / n, -1.0f / n};
    for (int t = 0; t < MAX_RADIX_SIZE; t++) {
      read_buf[i + t * m] *= inv_factor;
    }
  }

  // Copy everything in the complex case
  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    out[batch_idx + i + t * m] = read_buf[i + t * m];
  }

}

template <size_t n, size_t radix_2_steps, size_t radix_4_steps>
[[kernel]] void rfft(
    const device float* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {

  int batch_idx = thread_position_in_grid.x * n;
  int batch_idx_out = thread_position_in_grid.x * ((n/2) + 1);

  int i = thread_position_in_grid.y;
  int m = threads_per_grid.y;
 
  threadgroup float2 shared_in[n];
  threadgroup float2 shared_out[n];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // Copy input into shared memory
  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    shared_in[i + t * m].x = in[batch_idx + i + t * m];
  // Fill in the empty complex part in the shared buffer
    shared_in[i + t * m].y = 0;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, radix_2_steps, radix_4_steps, &read_buf, &write_buf);

  // For real to complex, we only need the first (n/2) + 1 terms
  // since the output is guaranteed to be hermitian symmetric
  for (int t = 0; t < MAX_RADIX_SIZE / 2; t++) {
    out[batch_idx_out + i + t * m] = read_buf[i + t * m];
  }
  // add on the +1 in (n/2) + 1
  if (i == 0) {
    out[batch_idx_out + MAX_RADIX_SIZE / 2 * m] = read_buf[MAX_RADIX_SIZE / 2 * m];
  }

}

template <size_t n, size_t radix_2_steps, size_t radix_4_steps>
[[kernel]] void irfft(
    const device float2* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {

  int batch_idx = thread_position_in_grid.x * ((n/2) + 1);
  int batch_idx_out = thread_position_in_grid.x * n;

  int i = thread_position_in_grid.y;
  int m = threads_per_grid.y;

  threadgroup float2 shared_in[n];
  threadgroup float2 shared_out[n];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // Copy the first n/2 + 1 inputs
  for (int t = 0; t < MAX_RADIX_SIZE / 2; t++) {
    shared_in[i + t * m] = in[batch_idx + i + t * m];
    // Conjugate since this is an inverse fft
    shared_in[i + t * m].y = -shared_in[i + t * m].y;
  }
  // add on the +1 in (n/2) + 1
  if (i == 0) {
    shared_in[MAX_RADIX_SIZE / 2 * m] = in[batch_idx + MAX_RADIX_SIZE / 2 * m];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Entries (n/2) + 1: are the reversed conjugates of the first half of the array
  for (int t = 0; t < MAX_RADIX_SIZE / 2; t++) {
    int index = i + t * m;
    shared_in[n - 1 - index] = shared_in[index + 1];
    shared_in[n - 1 - index].y = -shared_in[n - 1 - index].y;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, radix_2_steps, radix_4_steps, &read_buf, &write_buf);

  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    out[batch_idx_out + i + t * m] = read_buf[i + t * m].x / n;
  }

}

template <size_t n, bool inv, size_t radix_2_steps, size_t radix_4_steps>
[[kernel]] void bluestein_fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    const device float2* w_q [[buffer(2)]],
    const device float2* w_k [[buffer(3)]],
    constant const int& length,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
  // In numpy:
  // out = w_k * np.fft.ifft(np.fft.fft(w_k * in, n) * w_q)
  //
  // Where w_k and w_q are precomputed on CPU in high precision as:
  // w_k = np.exp(-1j * np.pi / n * (np.arange(-n + 1, n) ** 2))
  // w_q = np.fft.fft(1/w_k[-n:])

  int batch_idx = thread_position_in_grid.x * length;
  int i = thread_position_in_grid.y;
  int m = threads_per_grid.y;

  threadgroup float2 shared_in[n];
  threadgroup float2 shared_out[n];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // load input into shared memory
  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    int index = i + t * m;
    if (index < length) {
      float2 elem = in[batch_idx + index];
      if (inv) {
        elem.y = -elem.y;
      }
      shared_in[index] = complex_mul(elem, w_k[index]);
    } else {
      shared_in[index] = 0.0;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, radix_2_steps, radix_4_steps, &read_buf, &write_buf);

  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    int index = i + t * m;
    read_buf[index] = complex_mul(read_buf[index], w_q[index]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    read_buf[i + t * m].y = -read_buf[i + t * m].y;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, radix_2_steps, radix_4_steps, &read_buf, &write_buf);

  float2 inv_factor = {1.0f / n, -1.0f / n};
  float2 inv_factor_overall = {1.0f / length, -1.0f / length};

  for (int t = 0; t < MAX_RADIX_SIZE; t++) {
    int index = i + t * m;
    if (index < length) {
      float2 elem  = read_buf[index + length - 1] * inv_factor;
      elem = complex_mul(elem, w_k[index]);
      if (inv) {
        elem *= inv_factor_overall;
      }
      out[batch_idx + index] = elem;
    }
  }

}


#define instantiate_fft(name, n, inv, radix_2_steps, radix_4_steps) \
  template [[host_name("fft_" #name "_inv_" #inv)]] \
  [[kernel]] void fft<n, inv, radix_2_steps, radix_4_steps>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_rfft(name, n, radix_2_steps, radix_4_steps) \
  template [[host_name("rfft_" #name)]] \
  [[kernel]] void rfft<n, radix_2_steps, radix_4_steps>( \
      const device float* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_irfft(name, n, radix_2_steps, radix_4_steps) \
  template [[host_name("irfft_" #name)]] \
  [[kernel]] void irfft<n, radix_2_steps, radix_4_steps>( \
      const device float2* in [[buffer(0)]], \
      device float* out [[buffer(1)]], \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_bluestein(name, n, inv, radix_2_steps, radix_4_steps) \
  template [[host_name("bluestein_" #name "_inv_" #inv)]] \
  [[kernel]] void bluestein_fft<n, inv, radix_2_steps, radix_4_steps>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
      const device float2* w_q [[buffer(2)]], \
      const device float2* w_k [[buffer(2)]], \
    constant const int& length, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_ffts(name, n, radix_2_steps, radix_4_steps) \
    instantiate_fft(name, n, false, radix_2_steps, radix_4_steps) \
    instantiate_fft(name, n, true, radix_2_steps, radix_4_steps) \
    instantiate_rfft(name, n, radix_2_steps, radix_4_steps) \
    instantiate_irfft(name, n, radix_2_steps, radix_4_steps) \
    instantiate_bluestein(name, n, false, radix_2_steps, radix_4_steps) \
    instantiate_bluestein(name, n, true, radix_2_steps, radix_4_steps) \


// Explicitly define kernels for each power of 2.
instantiate_ffts(4, /* n= */ 4, /* radix_2_steps= */ 0, /* radix_4_steps= */ 1)
instantiate_ffts(8, 8, 1, 1)
instantiate_ffts(16, 16, 0, 2)
instantiate_ffts(32, 32, 1, 2)
instantiate_ffts(64, 64, 0, 3)
instantiate_ffts(128, 128, 1, 3)
instantiate_ffts(256, 256, 0, 4)
instantiate_ffts(512, 512, 1, 4)
instantiate_ffts(1024, 1024, 0, 5)
// 2048 is the max that will fit into 32KB of threadgroup memory.
// TODO: implement 4 step FFT for larger n.
instantiate_ffts(2048, 2048, 1, 5)
