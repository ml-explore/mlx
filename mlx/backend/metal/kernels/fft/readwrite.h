// Copyright © 2024 Apple Inc.

#include <metal_common>

#include "mlx/backend/metal/kernels/fft/radix.h"

/* FFT helpers for reading and writing from/to device memory.

For many sizes, GPU FFTs are memory bandwidth bound so
read/write performance is important.

Where possible, we read 128 bits sequentially in each thread,
coalesced with accesses from adajcent threads for optimal performance.

We implement specialized reading/writing for:
  - FFT
  - RFFT
  - IRFFT

Each with support for:
  - Contiguous reads
  - Padded reads
  - Strided reads
*/

#define MAX_RADIX 13

using namespace metal;

template <
    typename in_T,
    typename out_T,
    int step = 0,
    bool four_step_real = false>
struct ReadWriter {
  const device in_T* in;
  threadgroup float2* buf;
  device out_T* out;
  int n;
  int batch_size;
  int elems_per_thread;
  uint3 elem;
  uint3 grid;
  int threads_per_tg;
  bool inv;

  // Used for strided access
  int strided_device_idx = 0;
  int strided_shared_idx = 0;

  METAL_FUNC ReadWriter(
      const device in_T* in_,
      threadgroup float2* buf_,
      device out_T* out_,
      const short n_,
      const int batch_size_,
      const short elems_per_thread_,
      const uint3 elem_,
      const uint3 grid_,
      const bool inv_)
      : in(in_),
        buf(buf_),
        out(out_),
        n(n_),
        batch_size(batch_size_),
        elems_per_thread(elems_per_thread_),
        elem(elem_),
        grid(grid_),
        inv(inv_) {
    // Account for padding on last threadgroup
    threads_per_tg = elem.x == grid.x - 1
        ? (batch_size - (grid.x - 1) * grid.y) * grid.z
        : grid.y * grid.z;
  }

  // ifft(x) = 1/n * conj(fft(conj(x)))
  METAL_FUNC float2 post_in(float2 elem) const {
    return inv ? float2(elem.x, -elem.y) : elem;
  }

  // Handle float case for generic RFFT alg
  METAL_FUNC float2 post_in(float elem) const {
    return float2(elem, 0);
  }

  METAL_FUNC float2 pre_out(float2 elem) const {
    return inv ? float2(elem.x / n, -elem.y / n) : elem;
  }

  METAL_FUNC float2 pre_out(float2 elem, int length) const {
    return inv ? float2(elem.x / length, -elem.y / length) : elem;
  }

  METAL_FUNC bool out_of_bounds() const {
    // Account for possible extra threadgroups
    int grid_index = elem.x * grid.y + elem.y;
    return grid_index >= batch_size;
  }

  METAL_FUNC void load() const {
    int batch_idx = elem.x * grid.y * n;
    short tg_idx = elem.y * grid.z + elem.z;
    short max_index = grid.y * n - 2;

    // 2 complex64s = 128 bits
    constexpr int read_width = 2;
    for (short e = 0; e < (elems_per_thread / read_width); e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      index = metal::min(index, max_index);
      // vectorized reads
      buf[index] = post_in(in[batch_idx + index]);
      buf[index + 1] = post_in(in[batch_idx + index + 1]);
    }
    max_index += 1;
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      index = metal::min(index, max_index);
      buf[index] = post_in(in[batch_idx + index]);
    }
  }

  METAL_FUNC void write() const {
    int batch_idx = elem.x * grid.y * n;
    short tg_idx = elem.y * grid.z + elem.z;
    short max_index = grid.y * n - 2;

    constexpr int read_width = 2;
    for (short e = 0; e < (elems_per_thread / read_width); e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      index = metal::min(index, max_index);
      // vectorized reads
      out[batch_idx + index] = pre_out(buf[index]);
      out[batch_idx + index + 1] = pre_out(buf[index + 1]);
    }
    max_index += 1;
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      index = metal::min(index, max_index);
      out[batch_idx + index] = pre_out(buf[index]);
    }
  }

  // Padded IO for Bluestein's algorithm
  METAL_FUNC void load_padded(int length, const device float2* w_k) const {
    int batch_idx = elem.x * grid.y * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;

    threadgroup float2* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = metal::min(fft_idx + e * m, n - 1);
      if (index < length) {
        float2 elem = post_in(in[batch_idx + index]);
        seq_buf[index] = complex_mul(elem, w_k[index]);
      } else {
        seq_buf[index] = 0.0;
      }
    }
  }

  METAL_FUNC void write_padded(int length, const device float2* w_k) const {
    int batch_idx = elem.x * grid.y * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;
    float2 inv_factor = {1.0f / n, -1.0f / n};

    threadgroup float2* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = metal::min(fft_idx + e * m, n - 1);
      if (index < length) {
        float2 elem = seq_buf[index + length - 1] * inv_factor;
        out[batch_idx + index] = pre_out(complex_mul(elem, w_k[index]), length);
      }
    }
  }

  // Strided IO for four step FFT
  METAL_FUNC void compute_strided_indices(int stride, int overall_n) {
    // Use the batch threadgroup dimension to coalesce memory accesses:
    // e.g. stride = 12
    // device      | shared mem
    // 0  1  2  3  |  0 12 - -
    // -  -  -  -  |  1 13 - -
    // -  -  -  -  |  2 14 - -
    // 12 13 14 15 |  3 15 - -
    int coalesce_width = grid.y;
    int tg_idx = elem.y * grid.z + elem.z;
    int outer_batch_size = stride / coalesce_width;

    int strided_batch_idx = (elem.x % outer_batch_size) * coalesce_width +
        overall_n * (elem.x / outer_batch_size);
    strided_device_idx = strided_batch_idx +
        tg_idx / coalesce_width * elems_per_thread * stride +
        tg_idx % coalesce_width;
    strided_shared_idx = (tg_idx % coalesce_width) * n +
        tg_idx / coalesce_width * elems_per_thread;
  }

  // Four Step FFT First Step
  METAL_FUNC void load_strided(int stride, int overall_n) {
    compute_strided_indices(stride, overall_n);
    for (int e = 0; e < elems_per_thread; e++) {
      buf[strided_shared_idx + e] =
          post_in(in[strided_device_idx + e * stride]);
    }
  }

  METAL_FUNC void write_strided(int stride, int overall_n) {
    for (int e = 0; e < elems_per_thread; e++) {
      float2 output = buf[strided_shared_idx + e];
      int combined_idx = (strided_device_idx + e * stride) % overall_n;
      int ij = (combined_idx / stride) * (combined_idx % stride);
      // Apply four step twiddles at end of first step
      float2 twiddle = get_twiddle(ij, overall_n);
      out[strided_device_idx + e * stride] = complex_mul(output, twiddle);
    }
  }
};

// Four Step FFT Second Step
template <>
METAL_FUNC void ReadWriter<float2, float2, /*step=*/1>::load_strided(
    int stride,
    int overall_n) {
  // Silence compiler warnings
  (void)stride;
  (void)overall_n;
  // Don't invert between steps
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}

template <>
METAL_FUNC void ReadWriter<float2, float2, /*step=*/1>::write_strided(
    int stride,
    int overall_n) {
  compute_strided_indices(stride, overall_n);
  for (int e = 0; e < elems_per_thread; e++) {
    float2 output = buf[strided_shared_idx + e];
    out[strided_device_idx + e * stride] = pre_out(output, overall_n);
  }
}

// For RFFT, we interleave batches of two real sequences into one complex one:
//
// z_k = x_k + j.y_k
// X_k = (Z_k + Z_(N-k)*) / 2
// Y_k = -j * ((Z_k - Z_(N-k)*) / 2)
//
// This roughly doubles the throughput over the regular FFT.
template <>
METAL_FUNC bool ReadWriter<float, float2>::out_of_bounds() const {
  int grid_index = elem.x * grid.y + elem.y;
  // We pack two sequences into one for RFFTs
  return grid_index * 2 >= batch_size;
}

template <>
METAL_FUNC void ReadWriter<float, float2>::load() const {
  int batch_idx = elem.x * grid.y * n * 2 + elem.y * n * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    seq_buf[index].x = in[batch_idx + index];
    seq_buf[index].y = in[batch_idx + index + next_in];
  }
}

template <>
METAL_FUNC void ReadWriter<float, float2>::write() const {
  short n_over_2 = (n / 2) + 1;

  int batch_idx = elem.x * grid.y * n_over_2 * 2 + elem.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n_over_2;

  float2 conj = {1, -1};
  float2 minus_j = {0, -1};

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread / 2 + 1; e++) {
    int index = metal::min(fft_idx + e * m, n_over_2 - 1);
    // x_0 = z_0.real
    // y_0 = z_0.imag
    if (index == 0) {
      out[batch_idx + index] = {seq_buf[index].x, 0};
      out[batch_idx + index + next_out] = {seq_buf[index].y, 0};
    } else {
      float2 x_k = seq_buf[index];
      float2 x_n_minus_k = seq_buf[n - index] * conj;
      out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx + index + next_out] =
          complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float, float2>::load_padded(
    int length,
    const device float2* w_k) const {
  int batch_idx = elem.x * grid.y * length * 2 + elem.y * length * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : length;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    if (index < length) {
      float2 elem =
          float2(in[batch_idx + index], in[batch_idx + index + next_in]);
      seq_buf[index] = complex_mul(elem, w_k[index]);
    } else {
      seq_buf[index] = 0;
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float, float2>::write_padded(
    int length,
    const device float2* w_k) const {
  int length_over_2 = (length / 2) + 1;
  int batch_idx =
      elem.x * grid.y * length_over_2 * 2 + elem.y * length_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n + length - 1;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out = batch_size % 2 == 1 && grid_index * 2 == batch_size - 1
      ? 0
      : length_over_2;

  float2 conj = {1, -1};
  float2 inv_factor = {1.0f / n, -1.0f / n};
  float2 minus_j = {0, -1};

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread / 2 + 1; e++) {
    int index = metal::min(fft_idx + e * m, length_over_2 - 1);
    // x_0 = z_0.real
    // y_0 = z_0.imag
    if (index == 0) {
      float2 elem = complex_mul(w_k[index], seq_buf[index] * inv_factor);
      out[batch_idx + index] = float2(elem.x, 0);
      out[batch_idx + index + next_out] = float2(elem.y, 0);
    } else {
      float2 x_k = complex_mul(w_k[index], seq_buf[index] * inv_factor);
      float2 x_n_minus_k = complex_mul(
          w_k[length - index], seq_buf[length - index] * inv_factor);
      x_n_minus_k *= conj;
      // w_k should happen before this extraction
      out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx + index + next_out] =
          complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
}

// For IRFFT, we do the opposite
//
// Z_k = X_k + j.Y_k
// x_k = Re(Z_k)
// Y_k = Imag(Z_k)
template <>
METAL_FUNC bool ReadWriter<float2, float>::out_of_bounds() const {
  int grid_index = elem.x * grid.y + elem.y;
  // We pack two sequences into one for IRFFTs
  return grid_index * 2 >= batch_size;
}

template <>
METAL_FUNC void ReadWriter<float2, float>::load() const {
  short n_over_2 = (n / 2) + 1;
  int batch_idx = elem.x * grid.y * n_over_2 * 2 + elem.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n_over_2;

  short m = grid.z;
  short fft_idx = elem.z;

  float2 conj = {1, -1};
  float2 plus_j = {0, 1};

  for (int t = 0; t < elems_per_thread / 2 + 1; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
    float2 x = in[batch_idx + index];
    float2 y = in[batch_idx + index + next_in];
    // NumPy forces first input to be real
    bool first_val = index == 0;
    // NumPy forces last input on even irffts to be real
    bool last_val = n % 2 == 0 && index == n_over_2 - 1;
    if (first_val || last_val) {
      x = float2(x.x, 0);
      y = float2(y.x, 0);
    }
    seq_buf[index] = x + complex_mul(y, plus_j);
    seq_buf[index].y = -seq_buf[index].y;
    if (index > 0 && !last_val) {
      seq_buf[n - index] = (x * conj) + complex_mul(y * conj, plus_j);
      seq_buf[n - index].y = -seq_buf[n - index].y;
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float2, float>::write() const {
  int batch_idx = elem.x * grid.y * n * 2 + elem.y * n * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    out[batch_idx + index] = seq_buf[index].x / n;
    out[batch_idx + index + next_out] = seq_buf[index].y / -n;
  }
}

template <>
METAL_FUNC void ReadWriter<float2, float>::load_padded(
    int length,
    const device float2* w_k) const {
  int n_over_2 = (n / 2) + 1;
  int length_over_2 = (length / 2) + 1;

  int batch_idx =
      elem.x * grid.y * length_over_2 * 2 + elem.y * length_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in = batch_size % 2 == 1 && grid_index * 2 == batch_size - 1
      ? 0
      : length_over_2;

  short m = grid.z;
  short fft_idx = elem.z;

  float2 conj = {1, -1};
  float2 plus_j = {0, 1};

  for (int t = 0; t < elems_per_thread / 2 + 1; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
    float2 x = in[batch_idx + index];
    float2 y = in[batch_idx + index + next_in];
    if (index < length_over_2) {
      bool last_val = length % 2 == 0 && index == length_over_2 - 1;
      if (last_val) {
        x = float2(x.x, 0);
        y = float2(y.x, 0);
      }
      float2 elem1 = x + complex_mul(y, plus_j);
      seq_buf[index] = complex_mul(elem1 * conj, w_k[index]);
      if (index > 0 && !last_val) {
        float2 elem2 = (x * conj) + complex_mul(y * conj, plus_j);
        seq_buf[length - index] =
            complex_mul(elem2 * conj, w_k[length - index]);
      }
    } else {
      short pad_index = metal::min(length + (index - length_over_2) * 2, n - 2);
      seq_buf[pad_index] = 0;
      seq_buf[pad_index + 1] = 0;
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float2, float>::write_padded(
    int length,
    const device float2* w_k) const {
  int batch_idx = elem.x * grid.y * length * 2 + elem.y * length * 2;
  threadgroup float2* seq_buf = buf + elem.y * n + length - 1;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : length;

  short m = grid.z;
  short fft_idx = elem.z;

  float2 inv_factor = {1.0f / n, -1.0f / n};
  for (int e = 0; e < elems_per_thread; e++) {
    int index = fft_idx + e * m;
    if (index < length) {
      float2 output = complex_mul(seq_buf[index] * inv_factor, w_k[index]);
      out[batch_idx + index] = output.x / length;
      out[batch_idx + index + next_out] = output.y / -length;
    }
  }
}

// Four Step RFFT
template <>
METAL_FUNC void
ReadWriter<float2, float2, /*step=*/1, /*real=*/true>::load_strided(
    int stride,
    int overall_n) {
  // Silence compiler warnings
  (void)stride;
  (void)overall_n;
  // Don't invert between steps
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}

template <>
METAL_FUNC void
ReadWriter<float2, float2, /*step=*/1, /*real=*/true>::write_strided(
    int stride,
    int overall_n) {
  int overall_n_over_2 = overall_n / 2 + 1;
  int coalesce_width = grid.y;
  int tg_idx = elem.y * grid.z + elem.z;
  int outer_batch_size = stride / coalesce_width;

  int strided_batch_idx = (elem.x % outer_batch_size) * coalesce_width +
      overall_n_over_2 * (elem.x / outer_batch_size);
  strided_device_idx = strided_batch_idx +
      tg_idx / coalesce_width * elems_per_thread / 2 * stride +
      tg_idx % coalesce_width;
  strided_shared_idx = (tg_idx % coalesce_width) * n +
      tg_idx / coalesce_width * elems_per_thread / 2;
  for (int e = 0; e < elems_per_thread / 2; e++) {
    float2 output = buf[strided_shared_idx + e];
    out[strided_device_idx + e * stride] = output;
  }

  // Add on n/2 + 1 element
  if (tg_idx == 0 && elem.x % outer_batch_size == 0) {
    out[strided_batch_idx + overall_n / 2] = buf[n / 2];
  }
}

// Four Step IRFFT
template <>
METAL_FUNC void
ReadWriter<float2, float2, /*step=*/0, /*real=*/true>::load_strided(
    int stride,
    int overall_n) {
  int overall_n_over_2 = overall_n / 2 + 1;
  auto conj = float2(1, -1);

  compute_strided_indices(stride, overall_n);
  // Translate indices in terms of N - k
  for (int e = 0; e < elems_per_thread; e++) {
    int device_idx = strided_device_idx + e * stride;
    int overall_batch = device_idx / overall_n;
    int overall_index = device_idx % overall_n;
    if (overall_index < overall_n_over_2) {
      device_idx -= overall_batch * (overall_n - overall_n_over_2);
      buf[strided_shared_idx + e] = in[device_idx] * conj;
    } else {
      int conj_idx = overall_n - overall_index;
      device_idx = overall_batch * overall_n_over_2 + conj_idx;
      buf[strided_shared_idx + e] = in[device_idx];
    }
  }
}

template <>
METAL_FUNC void
ReadWriter<float2, float, /*step=*/1, /*real=*/true>::load_strided(
    int stride,
    int overall_n) {
  // Silence compiler warnings
  (void)stride;
  (void)overall_n;
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}

template <>
METAL_FUNC void
ReadWriter<float2, float, /*step=*/1, /*real=*/true>::write_strided(
    int stride,
    int overall_n) {
  compute_strided_indices(stride, overall_n);

  for (int e = 0; e < elems_per_thread; e++) {
    out[strided_device_idx + e * stride] =
        pre_out(buf[strided_shared_idx + e], overall_n).x;
  }
}
