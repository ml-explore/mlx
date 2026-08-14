// Copyright © 2024 Apple Inc.

// Metal FFT using Stockham's algorithm
//
// References:
// - VkFFT (https://github.com/DTolm/VkFFT)
// - Eric Bainville's excellent page (http://www.bealto.com/gpu-fft.html)

#include <metal_common>

#include "mlx/backend/metal/kernels/fft/radix.h"
#include "mlx/backend/metal/kernels/fft/readwrite.h"
#include "mlx/backend/metal/kernels/steel/defines.h"

using namespace metal;

#define MAX_RADIX 13
// Reached when elems_per_thread_ = 6, max_radix = 13
// and some threads have to do 3 radix 6s requiring 18 complex values.
#define MAX_OUTPUT_SIZE 18

// Specialize for a particular value of N at runtime
STEEL_CONST bool inv_ [[function_constant(0)]];
STEEL_CONST bool is_power_of_2_ [[function_constant(1)]];
STEEL_CONST int elems_per_thread_ [[function_constant(2)]];
// rader_m = n / rader_n
STEEL_CONST int rader_m_ [[function_constant(3)]];
// Stockham steps
STEEL_CONST int radix_13_steps_ [[function_constant(4)]];
STEEL_CONST int radix_11_steps_ [[function_constant(5)]];
STEEL_CONST int radix_8_steps_ [[function_constant(6)]];
STEEL_CONST int radix_7_steps_ [[function_constant(7)]];
STEEL_CONST int radix_6_steps_ [[function_constant(8)]];
STEEL_CONST int radix_5_steps_ [[function_constant(9)]];
STEEL_CONST int radix_4_steps_ [[function_constant(10)]];
STEEL_CONST int radix_3_steps_ [[function_constant(11)]];
STEEL_CONST int radix_2_steps_ [[function_constant(12)]];
// Rader steps
STEEL_CONST int rader_13_steps_ [[function_constant(13)]];
STEEL_CONST int rader_11_steps_ [[function_constant(14)]];
STEEL_CONST int rader_8_steps_ [[function_constant(15)]];
STEEL_CONST int rader_7_steps_ [[function_constant(16)]];
STEEL_CONST int rader_6_steps_ [[function_constant(17)]];
STEEL_CONST int rader_5_steps_ [[function_constant(18)]];
STEEL_CONST int rader_4_steps_ [[function_constant(19)]];
STEEL_CONST int rader_3_steps_ [[function_constant(20)]];
STEEL_CONST int rader_2_steps_ [[function_constant(21)]];
STEEL_CONST bool use_bluestein_twiddle_table_ [[function_constant(22)]];

// Generate the compact radix-8 Stockham twiddle table used by the fused
// 4096-point Bluestein convolution. The first p=1 stage needs no twiddles;
// the remaining stages contribute 8, 64, and 512 entries.
[[kernel]] void generate_bluestein_twiddles(
    device float2* twiddles [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  uint p;
  uint k;
  if (index < 8) {
    p = 8;
    k = index;
  } else if (index < 72) {
    p = 64;
    k = index - 8;
  } else {
    p = 512;
    k = index - 72;
  }
  twiddles[index] = get_twiddle<float>(k, 8 * p);
}

// See "radix.h" for radix codelets.
template <typename T>
using RadixFunc = void (*)(thread vec<T, 2>*, thread vec<T, 2>*);

// Perform a single radix n butterfly with appropriate twiddles
template <
    typename T,
    bool use_twiddle_table,
    int radix,
    RadixFunc<T> radix_func>
METAL_FUNC void radix_butterfly(
    int i,
    int p,
    int twiddle_offset,
    thread vec<T, 2>* x,
    thread short* indices,
    thread vec<T, 2>* y,
    const device vec<T, 2>* twiddles) {
  // i: the index in the overall DFT that we're processing.
  // p: the size of the DFTs we're merging at this step.
  // m: how many threads are working on this DFT.
  int k, j;

  // Use faster bitwise operations when working with powers of two
  constexpr bool radix_p_2 = (radix & (radix - 1)) == 0;
  if (radix_p_2 && is_power_of_2_) {
    constexpr short power = __builtin_ctz(radix);
    k = i & (p - 1);
    j = ((i - k) << power) + k;
  } else {
    k = i % p;
    j = (i / p) * radix * p + k;
  }

  // Apply twiddles
  if (p > 1) {
    vec<T, 2> twiddle_1;
    if constexpr (use_twiddle_table) {
      twiddle_1 = twiddles[twiddle_offset + k];
    } else {
      twiddle_1 = get_twiddle<T>(k, radix * p);
    }
    vec<T, 2> twiddle = twiddle_1;
    x[1] = complex_mul<T>(x[1], twiddle);

    STEEL_PRAGMA_UNROLL
    for (int t = 2; t < radix; t++) {
      twiddle = complex_mul<T>(twiddle, twiddle_1);
      x[t] = complex_mul<T>(x[t], twiddle);
    }
  }

  radix_func(x, y);

  STEEL_PRAGMA_UNROLL
  for (int t = 0; t < radix; t++) {
    indices[t] = j + t * p;
  }
}

// Perform all the radix steps required for a
// particular radix size n.
template <
    typename T,
    bool use_twiddle_table,
    int radix,
    RadixFunc<T> radix_func>
METAL_FUNC void radix_n_steps(
    int i,
    thread int* p,
    thread int* twiddle_offset,
    int m,
    int n,
    int num_steps,
    thread vec<T, 2>* inputs,
    thread short* indices,
    thread vec<T, 2>* values,
    threadgroup vec<T, 2>* buf,
    const device vec<T, 2>* twiddles) {
  int m_r = n / radix;
  // When combining different sized radices, we have to do
  // multiple butterflies in a single thread.
  // E.g. n = 28 = 4 * 7
  // 4 threads, 7 elems_per_thread
  // All threads do 1 radix7 butterfly.
  // 3 threads do 2 radix4 butterflies.
  // 1 thread does 1 radix4 butterfly.
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;

  int index = 0;
  int r_index = 0;
  for (int s = 0; s < num_steps; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          inputs[r] = buf[index + r * m_r];
        }
        radix_butterfly<T, use_twiddle_table, radix, radix_func>(
            index,
            *p,
            *twiddle_offset,
            inputs,
            indices + t * radix,
            values + t * radix,
            twiddles);
      }
    }

    // Wait until all threads have read their inputs into thread local mem
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          r_index = t * radix + r;
          buf[indices[r_index]] = values[r_index];
        }
      }
    }

    // Wait until all threads have written back to threadgroup mem
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (use_twiddle_table) {
      if (*p > 1) {
        *twiddle_offset += *p;
      }
    }
    *p *= radix;
  }
}

#define RADIX_STEP(radix, radix_func, num_steps)             \
  radix_n_steps<T, use_twiddle_table, radix, radix_func<T>>( \
      fft_idx,                                               \
      p,                                                     \
      &twiddle_offset,                                       \
      m,                                                     \
      n,                                                     \
      num_steps,                                             \
      inputs,                                                \
      indices,                                               \
      values,                                                \
      buf,                                                   \
      twiddles);

template <typename T, bool rader = false, bool use_twiddle_table = false>
METAL_FUNC void perform_fft(
    int fft_idx,
    thread int* p,
    int m,
    int n,
    threadgroup vec<T, 2>* buf,
    const device vec<T, 2>* twiddles = nullptr) {
  vec<T, 2> inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  vec<T, 2> values[MAX_OUTPUT_SIZE];
  int twiddle_offset = 0;

  RADIX_STEP(2, radix2, rader ? rader_2_steps_ : radix_2_steps_);
  RADIX_STEP(3, radix3, rader ? rader_3_steps_ : radix_3_steps_);
  RADIX_STEP(4, radix4, rader ? rader_4_steps_ : radix_4_steps_);
  RADIX_STEP(5, radix5, rader ? rader_5_steps_ : radix_5_steps_);
  RADIX_STEP(6, radix6, rader ? rader_6_steps_ : radix_6_steps_);
  RADIX_STEP(7, radix7, rader ? rader_7_steps_ : radix_7_steps_);
  RADIX_STEP(8, radix8, rader ? rader_8_steps_ : radix_8_steps_);
  RADIX_STEP(11, radix11, rader ? rader_11_steps_ : radix_11_steps_);
  RADIX_STEP(13, radix13, rader ? rader_13_steps_ : radix_13_steps_);
}

// Each FFT is computed entirely in shared GPU memory.
//
// N is decomposed into radix-n DFTs:
// e.g. 128 = 2 * 4 * 4 * 4
template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];

  thread ReadWriter<in_T, out_T> read_writer = ReadWriter<in_T, out_T>(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load();

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  int fft_idx = elem.z; // Thread index in DFT
  int m = grid.z; // Threads per DFT
  int tg_idx = elem.y * n; // Index of this DFT in threadgroup
  threadgroup vec<scalar_T, 2>* buf = &shared_in[tg_idx];

  perform_fft<scalar_T>(fft_idx, &p, m, n, buf);

  read_writer.write();
}

template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void rader_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    const device vec<typename FFTIOTypeTraits<in_T, out_T>::scalar_T, 2>*
        raders_b_q [[buffer(2)]],
    const device short* raders_g_q [[buffer(3)]],
    const device short* raders_g_minus_q [[buffer(4)]],
    constant const int& n,
    constant const int& batch_size,
    constant const int& rader_n,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  // Use Rader's algorithm to compute fast FFTs
  // when a prime factor `p` of `n` is greater than 13 but
  // has `p - 1` Stockham decomposable into prime factors <= 13.
  //
  // E.g. n = 102
  //        = 2 * 3 * 17
  // .      = 2 * 3 * RADER(16)
  // .      = 2 * 3 * RADER(4 * 4)
  //
  // In numpy:
  //   x_perm = x[g_q]
  //   y = np.fft.fft(x_perm) * b_q
  //   z = np.fft.ifft(y) + x[0]
  //   out = z[g_minus_q]
  //   out[0]  = x[1:].sum()
  //
  // Where the g_q and g_minus_q are permutations formed
  // by the group under multiplicative modulo N using the
  // primitive root of N and b_q is a constant.
  // See https://en.wikipedia.org/wiki/Rader%27s_FFT_algorithm
  //
  // Rader's uses fewer operations than Bluestein's and so
  // is more accurate. It's also faster in most cases.
  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];

  thread ReadWriter<in_T, out_T> read_writer = ReadWriter<in_T, out_T>(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load();

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The number of the threads we're using for each DFT
  int m = grid.z;

  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  threadgroup vec<scalar_T, 2>* buf = &shared_in[tg_idx];

  // rader_m = n / rader_n;
  int rader_m = rader_m_;

  // We have to load two x_0s for each thread since sometimes
  // elems_per_thread_ crosses a boundary.
  // E.g. with n = 34, rader_n = 17, elems_per_thread_ = 4
  // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8
  // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  short x_0_index =
      metal::min(fft_idx * elems_per_thread_ / (rader_n - 1), rader_m - 1);
  vec<scalar_T, 2> x_0[2] = {buf[x_0_index], buf[x_0_index + 1]};

  // Do the Rader permutation in shared memory
  vec<scalar_T, 2> temp[MAX_RADIX];
  int max_index = n - rader_m - 1;
  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short g_q = raders_g_q[index / rader_m];
    temp[e] = buf[rader_m + (g_q - 1) * rader_m + index % rader_m];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    buf[index + rader_m] = temp[e];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Rader FFT on x[rader_m:]
  int p = 1;
  perform_fft<scalar_T, /*rader=*/true>(
      fft_idx, &p, m, n - rader_m, buf + rader_m);

  // x_1 + ... + x_n is computed for us in the first FFT step so
  // we save it in the first rader_m indices of the array for later.
  int x_sum_index = metal::min(fft_idx, rader_m - 1);
  buf[x_sum_index] = buf[rader_m + x_sum_index * (rader_n - 1)];

  // The convolution is conjugated and scaled by 1/(rader_n - 1) exactly once
  // each. Float lanes apply the scale after the inverse FFT, preserving the
  // existing float code generation; reduced lanes apply it before, so the
  // intermediates cannot overflow the narrow lane range.
  scalar_T rader_inv_r = static_cast<scalar_T>(1.0f / (rader_n - 1));
  vec<scalar_T, 2> convolution_factor = {1.0f, -1.0f};
  vec<scalar_T, 2> convolution_inv_factor = {rader_inv_r, -rader_inv_r};
  if constexpr (!metal::is_same_v<scalar_T, float>) {
    convolution_factor = convolution_inv_factor;
    convolution_inv_factor = {1.0f, -1.0f};
  }

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short interleaved_index =
        index / rader_m + (index % rader_m) * (rader_n - 1);
    temp[e] = complex_mul<scalar_T>(
        buf[rader_m + interleaved_index],
        raders_b_q[interleaved_index % (rader_n - 1)]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    buf[rader_m + index] = temp[e] * convolution_factor;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Rader IFFT on x[rader_m:]
  p = 1;
  perform_fft<scalar_T, /*rader=*/true>(
      fft_idx, &p, m, n - rader_m, buf + rader_m);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, n - rader_m - 1);
    short diff_index = index / (rader_n - 1) - x_0_index;
    temp[e] = buf[rader_m + index] * convolution_inv_factor + x_0[diff_index];
  }

  // Use the sum of elements that was computed in the first FFT
  vec<scalar_T, 2> x_sum = buf[x_0_index] + x_0[0];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short g_q_index = index % (rader_n - 1);
    short g_q = raders_g_minus_q[g_q_index];
    short out_index = index - g_q_index + g_q + (index / (rader_n - 1));
    buf[out_index] = temp[e];
  }

  buf[x_0_index * rader_n] = x_sum;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  p = rader_n;
  perform_fft<scalar_T>(fft_idx, &p, m, n, buf);

  read_writer.write();
}

template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void bluestein_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    const device vec<typename FFTIOTypeTraits<in_T, out_T>::scalar_T, 2>* w_q
    [[buffer(2)]],
    const device vec<typename FFTIOTypeTraits<in_T, out_T>::scalar_T, 2>* w_k
    [[buffer(3)]],
    constant const int& length,
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  // Computes arbitrary length FFTs with Bluestein's algorithm
  //
  // In numpy:
  //   bluestein_n = next_power_of_2(2*n - 1)
  //   out = w_k * np.fft.ifft(np.fft.fft(w_k * in, bluestein_n) * w_q)
  //
  // Where w_k and w_q are precomputed on CPU in high precision as:
  //   w_k = np.exp(-1j * np.pi / n * (np.arange(-n + 1, n) ** 2))
  //   w_q = np.fft.fft(1/w_k[-n:])
  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];

  thread ReadWriter<in_T, out_T> read_writer = ReadWriter<in_T, out_T>(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load_padded(length, w_k);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  int fft_idx = elem.z; // Thread index in DFT
  int m = grid.z; // Threads per DFT
  int tg_idx = elem.y * n; // Index of this DFT in threadgroup
  threadgroup vec<scalar_T, 2>* buf = &shared_in[tg_idx];
  const device vec<scalar_T, 2>* twiddles = w_q + n;

  // fft
  if constexpr (
      tg_mem_size == 4096 && metal::is_same_v<in_T, float2> &&
      metal::is_same_v<out_T, float2>) {
    if (use_bluestein_twiddle_table_) {
      perform_fft<scalar_T, false, true>(fft_idx, &p, m, n, buf, twiddles);
    } else {
      perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
    }
  } else {
    perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
  }

  vec<scalar_T, 2> convolution_factor = {1.0f, -1.0f};
  if constexpr (!metal::is_same_v<scalar_T, float>) {
    // Reduced lanes normalize the convolution before the inverse FFT so the
    // intermediates cannot overflow; the write path then only conjugates.
    scalar_T inv_n = static_cast<scalar_T>(1.0f / n);
    convolution_factor = {inv_n, -inv_n};
  }
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    buf[index] =
        complex_mul<scalar_T>(buf[index], w_q[index]) * convolution_factor;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  p = 1;
  if constexpr (
      tg_mem_size == 4096 && metal::is_same_v<in_T, float2> &&
      metal::is_same_v<out_T, float2>) {
    if (use_bluestein_twiddle_table_) {
      perform_fft<scalar_T, false, true>(fft_idx, &p, m, n, buf, twiddles);
    } else {
      perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
    }
  } else {
    perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
  }

  read_writer.write_padded(length, w_k);
}

template <
    int tg_mem_size,
    typename in_T,
    typename out_T,
    int step,
    bool real = false>
[[kernel]] void four_step_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    constant const int& n1,
    constant const int& n2,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  // Fast four step FFT implementation for powers of 2.
  int overall_n = n1 * n2;
  int n = step == 0 ? n1 : n2;
  int stride = step == 0 ? n2 : n1;

  // The number of the threads we're using for each DFT
  int m = grid.z;
  int fft_idx = elem.z;

  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];
  threadgroup vec<scalar_T, 2>* buf = &shared_in[elem.y * n];

  using read_writer_t = ReadWriter<in_T, out_T, step, real>;
  read_writer_t read_writer = read_writer_t(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load_strided(stride, overall_n);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  perform_fft<scalar_T>(fft_idx, &p, m, n, buf);

  read_writer.write_strided(stride, overall_n);
}
