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

// OK so here's the plan:
// - Fix RFFT
// - Basic 4 step implementation
// - Optimize 7/11/13
// - Clean up and prepare PR

// Specialize for a particular value of N at runtime
constant bool inv_ [[function_constant(0)]];
constant bool is_power_of_2_ [[function_constant(1)]];
constant int elems_per_thread_ [[function_constant(2)]];
constant int radix_7_steps_ [[function_constant(3)]];
constant int radix_5_steps_ [[function_constant(4)]];
constant int radix_4_steps_ [[function_constant(5)]];
constant int radix_3_steps_ [[function_constant(6)]];
constant int radix_2_steps_ [[function_constant(7)]];

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
  float2 w_0 = {1.0, 0.0};
  float2 w_1 = {-0.49999999999999983, -0.8660254037844387};
  float2 w_2 = {-0.5000000000000004, 0.8660254037844384};

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + 1*m];
  float2 x_2 = read_buf[i + 2*m];

  int k = i % p;
  int j = (i / p) * 3 * p + k;

  float2 twiddle_1 = get_twiddle(k, 3*p);
  float2 twiddle_2 = complex_mul(twiddle_1, twiddle_1);

  x_1 = complex_mul(x_1, twiddle_1);
  x_2 = complex_mul(x_2, twiddle_2);

  float2 y_0 = x_0 + complex_mul(x_1, w_0) + complex_mul(x_2, w_0);
  float2 y_1 = x_0 + complex_mul(x_1, w_1) + complex_mul(x_2, w_2);
  float2 y_2 = x_0 + complex_mul(x_1, w_2) + complex_mul(x_2, w_1);

  write_buf[j] = y_0;
  write_buf[j + 1*p] = y_1;
  write_buf[j + 2*p] = y_2;
}

void radix4(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2*m];
  float2 x_3 = read_buf[i + 3*m];

  // We use faster bit shifting ops when n is a power of 2
  int k;
  if (is_power_of_2_) {
    k = i & (p - 1);
  } else {
    k = i % p;
  }

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

  int j;
  if (is_power_of_2_) {
    j = ((i - k) << 2) + k;
  } else {
    j = (i / p) * 4 * p + k;
  }

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
  write_buf[j + 2*p] = y_2;
  write_buf[j + 3*p] = y_3;
}

void radix5(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 w_0 = {1.0, 0.0};
  float2 w_1 = {0.30901699437494745, -0.9510565162951535};
  float2 w_2 = {-0.8090169943749473, -0.5877852522924732};
  float2 w_3 = {-0.8090169943749475, 0.587785252292473};
  float2 w_4 = {0.30901699437494723, 0.9510565162951536};

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + 1*m];
  float2 x_2 = read_buf[i + 2*m];
  float2 x_3 = read_buf[i + 3*m];
  float2 x_4 = read_buf[i + 4*m];

  int k = i % p;
  int j = (i / p) * 5 * p + k;

  float2 twiddle_1 = get_twiddle(k, 5*p);
  float2 twiddle_2 = complex_mul(twiddle_1, twiddle_1);
  float2 twiddle_3 = complex_mul(twiddle_1, twiddle_2);
  float2 twiddle_4 = complex_mul(twiddle_1, twiddle_3);

  x_1 = complex_mul(x_1, twiddle_1);
  x_2 = complex_mul(x_2, twiddle_2);
  x_3 = complex_mul(x_3, twiddle_3);
  x_4 = complex_mul(x_4, twiddle_4);

  float2 y_0 = x_0 + complex_mul(x_1, w_0) + complex_mul(x_2, w_0) + complex_mul(x_3, w_0) + complex_mul(x_4, w_0);
  float2 y_1 = x_0 + complex_mul(x_1, w_1) + complex_mul(x_2, w_2) + complex_mul(x_3, w_3) + complex_mul(x_4, w_4);
  float2 y_2 = x_0 + complex_mul(x_1, w_2) + complex_mul(x_2, w_4) + complex_mul(x_3, w_1) + complex_mul(x_4, w_3);
  float2 y_3 = x_0 + complex_mul(x_1, w_3) + complex_mul(x_2, w_1) + complex_mul(x_3, w_4) + complex_mul(x_4, w_2);
  float2 y_4 = x_0 + complex_mul(x_1, w_4) + complex_mul(x_2, w_3) + complex_mul(x_3, w_2) + complex_mul(x_4, w_1);

  write_buf[j] = y_0;
  write_buf[j + 1*p] = y_1;
  write_buf[j + 2*p] = y_2;
  write_buf[j + 3*p] = y_3;
  write_buf[j + 4*p] = y_4;
}

void radix7(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 w_0 = {1.0, 0.0};
  float2 w_1 = {0.6234898018587336, -0.7818314824680298};
  float2 w_2 = {-0.22252093395631434, -0.9749279121818236};
  float2 w_3 = {-0.900968867902419, -0.43388373911755823};
  float2 w_4 = {-0.9009688679024191, 0.433883739117558};
  float2 w_5 = {-0.2225209339563146, 0.9749279121818236};
  float2 w_6 = {0.6234898018587334, 0.7818314824680299};

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + 1*m];
  float2 x_2 = read_buf[i + 2*m];
  float2 x_3 = read_buf[i + 3*m];
  float2 x_4 = read_buf[i + 4*m];
  float2 x_5 = read_buf[i + 5*m];
  float2 x_6 = read_buf[i + 6*m];

  int k = i % p;
  int j = (i / p) * 7 * p + k;

  float2 twiddle_1 = get_twiddle(k, 7*p);
  float2 twiddle_2 = complex_mul(twiddle_1, twiddle_1);
  float2 twiddle_3 = complex_mul(twiddle_1, twiddle_2);
  float2 twiddle_4 = complex_mul(twiddle_1, twiddle_3);
  float2 twiddle_5 = complex_mul(twiddle_1, twiddle_4);
  float2 twiddle_6 = complex_mul(twiddle_1, twiddle_5);

  x_1 = complex_mul(x_1, twiddle_1);
  x_2 = complex_mul(x_2, twiddle_2);
  x_3 = complex_mul(x_3, twiddle_3);
  x_4 = complex_mul(x_4, twiddle_4);
  x_5 = complex_mul(x_5, twiddle_5);
  x_6 = complex_mul(x_6, twiddle_6);

  float2 y_0 = x_0 + complex_mul(x_1, w_0) + complex_mul(x_2, w_0) + complex_mul(x_3, w_0) + complex_mul(x_4, w_0) + complex_mul(x_5, w_0) + complex_mul(x_6, w_0);
  float2 y_1 = x_0 + complex_mul(x_1, w_1) + complex_mul(x_2, w_2) + complex_mul(x_3, w_3) + complex_mul(x_4, w_4) + complex_mul(x_5, w_5) + complex_mul(x_6, w_6);
  float2 y_2 = x_0 + complex_mul(x_1, w_2) + complex_mul(x_2, w_4) + complex_mul(x_3, w_6) + complex_mul(x_4, w_1) + complex_mul(x_5, w_3) + complex_mul(x_6, w_5);
  float2 y_3 = x_0 + complex_mul(x_1, w_3) + complex_mul(x_2, w_6) + complex_mul(x_3, w_2) + complex_mul(x_4, w_5) + complex_mul(x_5, w_1) + complex_mul(x_6, w_4);
  float2 y_4 = x_0 + complex_mul(x_1, w_4) + complex_mul(x_2, w_1) + complex_mul(x_3, w_5) + complex_mul(x_4, w_2) + complex_mul(x_5, w_6) + complex_mul(x_6, w_3);
  float2 y_5 = x_0 + complex_mul(x_1, w_5) + complex_mul(x_2, w_3) + complex_mul(x_3, w_1) + complex_mul(x_4, w_6) + complex_mul(x_5, w_4) + complex_mul(x_6, w_2);
  float2 y_6 = x_0 + complex_mul(x_1, w_6) + complex_mul(x_2, w_5) + complex_mul(x_3, w_4) + complex_mul(x_4, w_3) + complex_mul(x_5, w_2) + complex_mul(x_6, w_1);

  write_buf[j] = y_0;
  write_buf[j + 1*p] = y_1;
  write_buf[j + 2*p] = y_2;
  write_buf[j + 3*p] = y_3;
  write_buf[j + 4*p] = y_4;
  write_buf[j + 5*p] = y_5;
  write_buf[j + 6*p] = y_6;
}

void stockham_switch(threadgroup float2** read_buf, threadgroup float2** write_buf) {
    threadgroup float2* tmp = *write_buf;
    *write_buf = *read_buf;
    *read_buf = tmp;
}

void perform_fft(
    int i,  // thread index
    int n,  // overall fft size
    int m,  // total threads we have access to
    threadgroup float2** read_buf,
    threadgroup float2** write_buf) {
  int p = 1;

  int radix = 2;
  int m_r = n / radix;
  // ceil divide
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < radix_2_steps_; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      // Compiler will optimize this condition out for powers of 2 :)
      if (is_power_of_2_) {
        radix2(index, p, m_r, *read_buf, *write_buf);
      } else if (index < m_r) {
        radix2(index, p, m_r, *read_buf, *write_buf);
      }
    }
    p *= radix;

    stockham_switch(read_buf, write_buf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  radix = 3;
  m_r = n / radix;
  max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < radix_3_steps_; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        radix3(index, p, m_r, *read_buf, *write_buf);
      }
    }
    p *= radix;

    stockham_switch(read_buf, write_buf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  radix = 4;
  m_r = n / radix;
  max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < radix_4_steps_; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (is_power_of_2_) {
        radix4(index, p, m_r, *read_buf, *write_buf);
      } else if (index < m_r) {
        radix4(index, p, m_r, *read_buf, *write_buf);
      }
    }
    p *= radix;

    stockham_switch(read_buf, write_buf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  radix = 5;
  m_r = n / radix;
  max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < radix_5_steps_; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      radix5(index, p, m_r, *read_buf, *write_buf);
    }
    p *= radix;

    stockham_switch(read_buf, write_buf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  radix = 7;
  m_r = n / radix;
  max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < radix_7_steps_; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      radix7(index, p, m_r, *read_buf, *write_buf);
    }
    p *= radix;

    stockham_switch(read_buf, write_buf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    constant const int& batch_size,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {

  int fft_idx = thread_position_in_grid.z;
  int tg_idx = thread_position_in_grid.y * n;
  int batch_idx = thread_position_in_grid.x * threads_per_grid.y * n + tg_idx;

  // The number of the threads we're using for each DFT
  int m = threads_per_grid.z;

  // the thread group memory will be too big
  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];
  threadgroup float2* write_buf = &shared_out[tg_idx];

  // Account for possible extra threadgroups
  int overall_batch = thread_position_in_grid.x * threads_per_grid.y + thread_position_in_grid.y;
  if (overall_batch >= batch_size) {
    return;
  }

  // Copy input into shared memory
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    read_buf[index] = in[batch_idx + index];
    if (inv_) {
      read_buf[index].y = -read_buf[index].y;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(fft_idx, n, m, &read_buf, &write_buf);

  float2 inv_factor = {1.0f / n, -1.0f / n};
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    if (inv_) {
      read_buf[index] *= inv_factor;
    }
    out[batch_idx + index] = read_buf[index];
  }
}

template <int tg_mem_size>
[[kernel]] void rfft(
    const device float* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {

  int n_over_2 = (n/2) + 1;
  int fft_idx = thread_position_in_grid.z;
  int tg_idx = thread_position_in_grid.y * 2 * n;
  int batch_idx = thread_position_in_grid.x * threads_per_grid.y * 2 * n + tg_idx;
  int batch_idx_out = thread_position_in_grid.x * threads_per_grid.y * 
      2 * n_over_2 + thread_position_in_grid.y * 2 * n_over_2;

  int m = threads_per_grid.y;

  // Plan
  // - Fast RFFT (IRFFT can be slower for now)

  // Account for possible extra threadgroups
  int overall_batch = thread_position_in_grid.x * threads_per_grid.y + thread_position_in_grid.y;
  if (overall_batch >= batch_size) {
    return;
  }
 
  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = shared_in;
  threadgroup float2* write_buf = shared_out;

  // For rfft, interleave batches of two real sequences into one complex one (z).
  // Then compute the output as:
  // x_k = (Z_k + Z_k*) / 2
  // y_k = -j * ((Z_k - Z_k*) / 2)
  for (int t = 0; t < 1; t++) {
    int index = fft_idx + t * m;
    read_buf[index].x = in[batch_idx + index];
    // TODO: this needs to be padding if there's nothing here
    read_buf[index].y = in[batch_idx + index + n];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // perform_fft(fft_idx, n, m, &read_buf, &write_buf);

  float2 conj = {0, -1};
  for (int t = 0; t < 1; t++) {
    int index = fft_idx + t * m;
    // if (index == 0) {
    //   out[batch_idx_out + index] = read_buf[index].x;
    //   out[batch_idx_out + index + n] = read_buf[index].y;
    // }
    // float2 x_k = read_buf[index];
    // float2 x_n_minus_k = read_buf[n - index] * conj;
    out[batch_idx_out + index] = 9;
    // out[batch_idx_out + index + n_over_2] = 9;
    // out[batch_idx_out + index] = (x_k + x_n_minus_k) / 2;
    // out[batch_idx_out + index + n_over_2] = (x_k + x_n_minus_k) / 2;
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

  perform_fft(i, n, m, &read_buf, &write_buf);

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
    constant const int& batch_size,
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
  // Computes arbitrary length FFTs with Bluestein's algorithm
  // In numpy:
  // out = w_k * np.fft.ifft(np.fft.fft(w_k * in, n) * w_q)
  //
  // Where w_k and w_q are precomputed on CPU in high precision as:
  // w_k = np.exp(-1j * np.pi / n * (np.arange(-n + 1, n) ** 2))
  // w_q = np.fft.fft(1/w_k[-n:])

  int fft_idx = thread_position_in_grid.z;
  int tg_idx = thread_position_in_grid.y * n;
  int batch_idx = thread_position_in_grid.x * threads_per_grid.y * length + thread_position_in_grid.y * length;

  // Is this right? Should be if it's 9  3/
  int m = threads_per_grid.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];
  threadgroup float2* write_buf = &shared_out[tg_idx];

  // Account for possible extra threadgroups
  int overall_batch = thread_position_in_grid.x * threads_per_grid.y + thread_position_in_grid.y;
  if (overall_batch >= batch_size) {
    return;
  }

  // load input into shared memory
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    if (index < length) {
      float2 elem = in[batch_idx + index];
      if (inv_) {
        elem.y = -elem.y;
      }
      read_buf[index] = complex_mul(elem, w_k[index]);
    } else {
      read_buf[index] = 0.0;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(fft_idx, n, m, &read_buf, &write_buf);

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    read_buf[index] = complex_mul(read_buf[index], w_q[index]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  for (int t = 0; t < elems_per_thread_; t++) {
    read_buf[fft_idx + t * m].y = -read_buf[fft_idx + t * m].y;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  perform_fft(fft_idx, n, m, &read_buf, &write_buf);

  float2 inv_factor = {1.0f / n, -1.0f / n};
  float2 inv_factor_overall = {1.0f / length, -1.0f / length};

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
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
  template [[host_name("fft_mem_" #tg_mem_size)]] \
  [[kernel]] void fft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
    constant const int& n, \
    constant const int& batch_size, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_rfft(tg_mem_size) \
  template [[host_name("rfft_mem_" #tg_mem_size)]] \
  [[kernel]] void rfft<tg_mem_size>( \
      const device float* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
    constant const int& n, \
    constant const int& batch_size, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_irfft(tg_mem_size) \
  template [[host_name("irfft_mem_" #tg_mem_size)]] \
  [[kernel]] void irfft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float* out [[buffer(1)]], \
    constant const int& n, \
    uint3 thread_position_in_grid [[thread_position_in_grid]], \
    uint3 threads_per_grid [[threads_per_grid]]);

#define instantiate_bluestein(tg_mem_size) \
  template [[host_name("bluestein_fft_mem_" #tg_mem_size)]] \
  [[kernel]] void bluestein_fft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
      const device float2* w_q [[buffer(2)]], \
      const device float2* w_k [[buffer(2)]], \
    constant const int& length, \
    constant const int& n, \
    constant const int& batch_size, \
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
