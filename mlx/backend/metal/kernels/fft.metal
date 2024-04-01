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

// Specialize for a particular value of n at runtime
constant bool inv_ [[function_constant(0)]];
constant int elems_per_thread_ [[function_constant(1)]];
constant int radix_4_steps_ [[function_constant(2)]];
constant int radix_3_steps_ [[function_constant(3)]];
constant int radix_2_steps_ [[function_constant(4)]];

float2 complex_mul(float2 a, float2 b) {
  float2 c = {
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  };
  return c;
}

float2 get_twiddle(int k, int p) {
  float theta = -2.0f * k * M_PI_F / p;

  float2 twiddle = {
      metal::fast::cos(theta),
      metal::fast::sin(theta)
  };
  return twiddle;
}

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

void radix3(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  // Hardcoded twiddle factor for DFT3
  float2 w_1 = {-0.5f, -0.8660254037844387f};
  float2 w_2 = {-0.5f, 0.8660254037844387f};

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2*m];

  int k = i % p;

  float2 twiddle = get_twiddle(k, 3*p);
  float2 twiddle_2 = complex_mul(twiddle, twiddle);

  x_1 = complex_mul(x_1, twiddle);
  x_2 = complex_mul(x_2, twiddle_2);

  float2 y_0 = x_0 + x_1 + x_2;
  float2 y_1 = x_0 + complex_mul(x_1, w_1) + complex_mul(x_2, w_2);
  float2 y_2 = x_0 + complex_mul(x_1, w_2) + complex_mul(x_2, w_1);

  int j = (i / p) * 3 * p + k;

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
  write_buf[j + 2*p] = y_2;
}

void radix4(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2*m];
  float2 x_3 = read_buf[i + 3*m];

  // The index within this sub-DFT
  int k = i & (p - 1);

  float2 twiddle = get_twiddle(k, 4*p);
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
    threadgroup float2** read_buf,
    threadgroup float2** write_buf) {

  threadgroup float2* tmp;

  int p = 1;

  for (int r = 0; r < radix_2_steps_; r++) {
    radix2(i, p, m*2, *read_buf, *write_buf);
    radix2(i + m, p, m*2, *read_buf, *write_buf);
    p *= 2;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stockham switch of buffers
    tmp = *write_buf;
    *write_buf = *read_buf;
    *read_buf = tmp;
  }

  for (int r = 0; r < radix_4_steps_; r++) {
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
template <int tg_mem_size>
[[kernel]] void fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {

  // Index of the DFT in batch
  int batch_idx = thread_position_in_grid.x * n;
  // The index in the DFT we're working on
  int i = thread_position_in_grid.y;
  // The number of the threads we're using for each DFT
  int m = threads_per_grid.y;

  // Pick the closest shared memory size that we degine
  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // Copy input into shared memory
  for (int t = 0; t < elems_per_thread_; t++) {
    read_buf[i + t * m] = in[batch_idx + i + t * m];
  }

  // ifft(x) = (1/n)conj(fft(conj(x)))
  if (inv_) {
    for (int t = 0; t < elems_per_thread_; t++) {
      read_buf[i + t * m].y = -read_buf[i + t * m].y;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, &read_buf, &write_buf);

  if (inv_) {
    float2 inv_factor = {1.0f / n, -1.0f / n};
    for (int t = 0; t < elems_per_thread_; t++) {
      read_buf[i + t * m] *= inv_factor;
    }
  }

  for (int t = 0; t < elems_per_thread_; t++) {
    out[batch_idx + i + t * m] = read_buf[i + t * m];
  }

}

template <int tg_mem_size>
[[kernel]] void rfft(
    const device float* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {

  int batch_idx = thread_position_in_grid.x * n;
  int batch_idx_out = thread_position_in_grid.x * ((n/2) + 1);

  int i = thread_position_in_grid.y;
  int m = threads_per_grid.y;
 
  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // Copy input into shared memory
  for (int t = 0; t < elems_per_thread_; t++) {
    shared_in[i + t * m].x = in[batch_idx + i + t * m];
  // Fill in the empty complex part in the shared buffer
    shared_in[i + t * m].y = 0;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, &read_buf, &write_buf);

  // For real to complex, we only need the first (n/2) + 1 terms
  // since the output is guaranteed to be hermitian symmetric
  for (int t = 0; t < elems_per_thread_ / 2; t++) {
    out[batch_idx_out + i + t * m] = read_buf[i + t * m];
  }
  // add on the +1 in (n/2) + 1
  if (i == 0) {
    out[batch_idx_out + elems_per_thread_ / 2 * m] = read_buf[elems_per_thread_ / 2 * m];
  }

}

template <int tg_mem_size>
[[kernel]] void irfft(
    const device float2* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant const int& n,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {

  int batch_idx = thread_position_in_grid.x * ((n/2) + 1);
  int batch_idx_out = thread_position_in_grid.x * n;

  int i = thread_position_in_grid.y;
  int m = threads_per_grid.y;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // Copy the first n/2 + 1 inputs
  for (int t = 0; t < elems_per_thread_ / 2; t++) {
    shared_in[i + t * m] = in[batch_idx + i + t * m];
    // Conjugate since this is an inverse fft
    shared_in[i + t * m].y = -shared_in[i + t * m].y;
  }
  // add on the +1 in (n/2) + 1
  if (i == 0) {
    shared_in[elems_per_thread_ / 2 * m] = in[batch_idx + elems_per_thread_ / 2 * m];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Entries (n/2) + 1: are the reversed conjugates of the first half of the array
  for (int t = 0; t < elems_per_thread_ / 2; t++) {
    int index = i + t * m;
    shared_in[n - 1 - index] = shared_in[index + 1];
    shared_in[n- 1 - index].y = -shared_in[n - 1 - index].y;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, &read_buf, &write_buf);

  for (int t = 0; t < elems_per_thread_; t++) {
    out[batch_idx_out + i + t * m] = read_buf[i + t * m].x / n;
  }

}

template <int tg_mem_size>
[[kernel]] void bluestein_fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    const device float2* w_q [[buffer(2)]],
    const device float2* w_k [[buffer(3)]],
    constant const int& length,
    constant const int& n,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
  // Computes arbitrary length FFTs with Bluestein's algorithm
  // In numpy:
  // out = w_k * np.fft.ifft(np.fft.fft(w_k * in, n) * w_q)
  //
  // Where w_k and w_q are precomputed on CPU in high precision as:
  // w_k = np.exp(-1j * np.pi / n * (np.arange(-n + 1, n) ** 2))
  // w_q = np.fft.fft(1/w_k[-n:])

  int batch_idx = thread_position_in_grid.x * length;
  int i = thread_position_in_grid.y;
  int m = threads_per_grid.y;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // load input into shared memory
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = i + t * m;
    if (index < length) {
      float2 elem = in[batch_idx + index];
      if (inv_) {
        elem.y = -elem.y;
      }
      shared_in[index] = complex_mul(elem, w_k[index]);
    } else {
      shared_in[index] = 0.0;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, &read_buf, &write_buf);

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = i + t * m;
    read_buf[index] = complex_mul(read_buf[index], w_q[index]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  for (int t = 0; t < elems_per_thread_; t++) {
    read_buf[i + t * m].y = -read_buf[i + t * m].y;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(i, m, &read_buf, &write_buf);

  float2 inv_factor = {1.0f / n, -1.0f / n};
  float2 inv_factor_overall = {1.0f / length, -1.0f / length};

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = i + t * m;
    if (index < length) {
      float2 elem  = read_buf[index + length - 1] * inv_factor;
      elem = complex_mul(elem, w_k[index]);
      if (inv_) {
        elem *= inv_factor_overall;
      }
      out[batch_idx + index] = elem;
    }
  }
}


#define instantiate_fft(tg_mem_size) \
  template [[host_name("fft_" #tg_mem_size)]] \
  [[kernel]] void fft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
    constant const int& n, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_rfft(tg_mem_size) \
  template [[host_name("rfft_" #tg_mem_size)]] \
  [[kernel]] void rfft<tg_mem_size>( \
      const device float* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
    constant const int& n, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_irfft(tg_mem_size) \
  template [[host_name("irfft_" #tg_mem_size)]] \
  [[kernel]] void irfft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float* out [[buffer(1)]], \
    constant const int& n, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_bluestein(tg_mem_size) \
  template [[host_name("bluestein_" #tg_mem_size)]] \
  [[kernel]] void bluestein_fft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
      const device float2* w_q [[buffer(2)]], \
      const device float2* w_k [[buffer(2)]], \
    constant const int& length, \
    constant const int& n, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_ffts(tg_mem_size) \
    instantiate_fft(tg_mem_size) \
    instantiate_rfft(tg_mem_size) \
    instantiate_irfft(tg_mem_size) \
    instantiate_bluestein(tg_mem_size) \


// It's substantially faster to statically define the
// threadgroup memory size rather than using 
// `setThreadgroupMemoryLength` on the compute encoder.
// For non-power of 2 sizes we round up the shared memory.
instantiate_ffts(4)
instantiate_ffts(8)
instantiate_ffts(16)
instantiate_ffts(32)
instantiate_ffts(64)
instantiate_ffts(128)
instantiate_ffts(256)
instantiate_ffts(512)
instantiate_ffts(1024)
// 2048 is the max that will fit into 32KB of threadgroup memory.
instantiate_ffts(2048)
