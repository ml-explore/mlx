// Copyright © 2025 Apple Inc.
// NF4 (NormalFloat4) quantized matmul kernels for MLX.
//
// These kernels mirror the structure of fp_quantized.h but replace the
// E2M1 float dequantization with an NF4 lookup table, and read scales
// as float32 (4 bytes per group) instead of FP8 (1 byte per group).
//
// NF4 format: 4-bit indices into a 16-element LUT derived from normal
// distribution quantiles. Scales are float32 absmax per quantization group.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/nf4.h"

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int wsize = 8, int bits = 4>
inline constexpr short nf4_get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short nf4_get_bytes_per_pack() {
  return wsize / 8;
}

///////////////////////////////////////////////////////////////////////////////
// NF4-specific dequantization
///////////////////////////////////////////////////////////////////////////////

// Read a float32 scale from the uint8_t scale buffer.
// NF4 stores one float32 absmax per group (4 bytes per group).
template <typename T, int group_size>
static inline T nf4_dequantize_scale(const device uint8_t* s) {
  return T(*reinterpret_cast<const device float*>(s));
}

// Dequantize a 4-bit NF4 index to float via LUT lookup
template <typename U = float>
inline U nf4_dequantize_value(uint8_t x) {
  return U(nf4_lut[x & 0x0f]);
}

///////////////////////////////////////////////////////////////////////////////
// Vector load helpers (same as fp_quantized.h)
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, int values_per_thread>
inline void nf4_load_vector(const device T* x, thread U* x_thread) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    x_thread[i] = x[i];
  }
}

template <typename T, typename U, int values_per_thread>
inline void
nf4_load_vector_safe(const device T* x, thread U* x_thread, int N) {
  for (int i = 0; i < N; i++) {
    x_thread[i] = x[i];
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
}

///////////////////////////////////////////////////////////////////////////////
// NF4 dot product and outer product primitives
//
// Optimization: build a scaled LUT (nf4_lut[i] * absmax_scale) in thread
// registers, then the inner loop is just a register-local table lookup per
// nibble — no constant-memory access and no post-multiply needed.
///////////////////////////////////////////////////////////////////////////////

template <typename U>
inline void nf4_build_scaled_lut(U scale, thread U* slut) {
  // 16 entries, pre-multiplied by the block absmax scale.
  // The compiler should keep these in registers for the inner loop.
  slut[0]  = U(nf4_lut[0])  * scale;
  slut[1]  = U(nf4_lut[1])  * scale;
  slut[2]  = U(nf4_lut[2])  * scale;
  slut[3]  = U(nf4_lut[3])  * scale;
  slut[4]  = U(nf4_lut[4])  * scale;
  slut[5]  = U(nf4_lut[5])  * scale;
  slut[6]  = U(nf4_lut[6])  * scale;
  slut[7]  = U(nf4_lut[7])  * scale;
  slut[8]  = U(nf4_lut[8])  * scale;
  slut[9]  = U(nf4_lut[9])  * scale;
  slut[10] = U(nf4_lut[10]) * scale;
  slut[11] = U(nf4_lut[11]) * scale;
  slut[12] = U(nf4_lut[12]) * scale;
  slut[13] = U(nf4_lut[13]) * scale;
  slut[14] = U(nf4_lut[14]) * scale;
  slut[15] = U(nf4_lut[15]) * scale;
}

template <typename U, int values_per_thread>
inline U nf4_qdot(const device uint8_t* w, const thread U* x_thread, U scale) {
  // Build scaled LUT once per group
  thread U slut[16];
  nf4_build_scaled_lut(scale, slut);

  U accum = 0;
  // Process 2 elements per byte (lo nibble, hi nibble)
  for (int i = 0; i < (values_per_thread / 2); i++) {
    uint8_t byte = w[i];
    accum += x_thread[2 * i]     * slut[byte & 0x0f] +
             x_thread[2 * i + 1] * slut[byte >> 4];
  }
  return accum;  // no final multiply — already scaled in LUT
}

template <typename U, int values_per_thread>
inline U nf4_qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    int N) {
  thread U slut[16];
  nf4_build_scaled_lut(scale, slut);

  U accum = 0;
  for (int i = 0; i < (N / 2); i++) {
    uint8_t byte = w[i];
    accum += x_thread[2 * i]     * slut[byte & 0x0f] +
             x_thread[2 * i + 1] * slut[byte >> 4];
  }
  return accum;
}

template <typename U, int values_per_thread>
inline void
nf4_qouter(const thread uint8_t* w, U x, U scale, thread U* result) {
  // For outer product, pre-scale x instead of building full LUT
  thread U slut[16];
  nf4_build_scaled_lut(scale, slut);

  for (int i = 0; i < (values_per_thread / 2); i++) {
    result[2 * i]     += x * slut[w[i] & 0x0f];
    result[2 * i + 1] += x * slut[w[i] >> 4];
  }
}

template <typename U>
inline void nf4_dequantize_elem(uint8_t w, U scale, threadgroup U* w_local) {
  // Dequantize two 4-bit values from one byte
  w_local[0] = U(nf4_lut[w & 0x0f]) * scale;
  w_local[1] = U(nf4_lut[w >> 4]) * scale;
}

///////////////////////////////////////////////////////////////////////////////
// NF4 Block Loader
//
// Same structure as fp_quantized.h QuantizedBlockLoader but reads float32
// scales. Scale stride = sizeof(float) bytes per group instead of 1.
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size>
struct NF4BlockLoader {
  MLX_MTL_CONST short pack_factor = nf4_get_pack_factor();
  MLX_MTL_CONST short bytes_per_pack = nf4_get_bytes_per_pack();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps =
      group_size < BCOLS ? 1 : group_size / BCOLS;
  MLX_MTL_CONST short scale_step =
      group_size < BCOLS ? BCOLS / group_size : 1;

  // Scale stride in bytes: float32 = 4 bytes per scale
  MLX_MTL_CONST short scale_bytes = 4;

  static_assert(
      (n_reads * pack_factor) <= group_size,
      "The number of reads per thread must be less than the group size.");

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint8_t* scales;

  NF4BlockLoader(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(
            scales_ + (bi * src_ld / group_size +
                        (bj * pack_factor) / group_size) *
                scale_bytes) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = nf4_dequantize_scale<T, group_size>(scales);
    for (int i = 0; i < n_reads; i++) {
      nf4_dequantize_elem<T>(src[i * bytes_per_pack], scale, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = nf4_dequantize_scale<T, group_size>(scales);
    for (int i = 0; i < n_reads; i++) {
      nf4_dequantize_elem<T>(src[i * bytes_per_pack], scale, dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales += scale_bytes;
        }
      } else {
        scales += scale_step * scale_bytes;
      }
    } else {
      scales += group_stride * scale_bytes;
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// QMV kernels (matrix-vector, used during token generation)
///////////////////////////////////////////////////////////////////////////////

template <typename T, int group_size, int D>
METAL_FUNC void nf4_qmv_quad_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
  constexpr int pack_factor = nf4_get_pack_factor<32, 4>();
  constexpr int values_per_thread = D / QUAD_SIZE;
  constexpr int steps_per_thread =
      values_per_thread < group_size ? 1 : values_per_thread / group_size;
  constexpr int values_per_step = values_per_thread / steps_per_thread;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int packs_per_step = values_per_step / pack_factor;
  constexpr int results_per_quadgroup = 8;

  // NF4 scale stride: 4 bytes per float32 scale
  constexpr int scale_bytes = 4;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_quadgroup] = {0};

  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;

  w += out_row * in_vec_size_w + quad_lid * packs_per_thread;
  scales +=
      (out_row * in_vec_size_g + (quad_lid * values_per_thread) / group_size) *
      scale_bytes;
  x += tid.x * in_vec_size + quad_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  nf4_load_vector<T, U, values_per_thread>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(w + row * in_vec_size_w * quads_per_simd);
    const device uint8_t* sl =
        scales + row * in_vec_size_g * quads_per_simd * scale_bytes;
#pragma unroll
    for (int k = 0; k < steps_per_thread; ++k) {
      U s = nf4_dequantize_scale<U, group_size>(sl);
      if (row * quads_per_simd + out_row < out_vec_size) {
        result[row] +=
            nf4_qdot<U, values_per_step>(wl, x_thread + k * values_per_step, s);
      }
      sl += scale_bytes;
      wl += (sizeof(uint32_t) / sizeof(uint8_t)) * packs_per_step;
    }
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);
    if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
      y[row * quads_per_simd] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size>
METAL_FUNC void nf4_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = nf4_get_pack_factor<32, 4>();
  constexpr int bytes_per_pack = nf4_get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int scale_bytes = 4;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales +=
      (out_row * in_vec_size_g + simd_lid / scale_step_per_thread) *
      scale_bytes;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    nf4_load_vector<T, U, values_per_thread>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device uint8_t* sl =
          scales + row * in_vec_size_g * scale_bytes;

      U s = nf4_dequantize_scale<U, group_size>(sl);
      result[row] += nf4_qdot<U, values_per_thread>(wl, x_thread, s);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += (block_size / group_size) * scale_bytes;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size>
METAL_FUNC void nf4_qmv_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = nf4_get_pack_factor<32, 4>();
  constexpr int bytes_per_pack = nf4_get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int scale_bytes = 4;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales +=
        (out_row * in_vec_size_g + simd_lid / scale_step_per_thread) *
        scale_bytes;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      nf4_load_vector<T, U, values_per_thread>(x, x_thread);
      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device uint8_t* sl =
            scales + row * in_vec_size_g * scale_bytes;
        U s = nf4_dequantize_scale<U, group_size>(sl);
        result[row] += nf4_qdot<U, values_per_thread>(wl, x_thread, s);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      scales += (block_size / group_size) * scale_bytes;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      nf4_load_vector_safe<T, U, values_per_thread>(x, x_thread, remaining);
      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device uint8_t* sl =
            scales + row * in_vec_size_g * scale_bytes;
        U s = nf4_dequantize_scale<U, group_size>(sl);
        result[row] += nf4_qdot<U, values_per_thread>(wl, x_thread, s);
      }
    }
    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  } else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales +=
        (used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread) *
        scale_bytes;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      nf4_load_vector<T, U, values_per_thread>(x, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device uint8_t* sl =
            scales + row * in_vec_size_g * scale_bytes;
        U s = nf4_dequantize_scale<U, group_size>(sl);
        result[row] += nf4_qdot<U, values_per_thread>(wl, x_thread, s);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      scales += (block_size / group_size) * scale_bytes;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      nf4_load_vector_safe<T, U, values_per_thread>(x, x_thread, remaining);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device uint8_t* sl =
            scales + row * in_vec_size_g * scale_bytes;
        U s = nf4_dequantize_scale<U, group_size>(sl);
        result[row] +=
            nf4_qdot_safe<U, values_per_thread>(wl, x_thread, s, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// QVM kernel (vector-matrix)
///////////////////////////////////////////////////////////////////////////////

template <typename T, const int group_size>
METAL_FUNC void nf4_qvm_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    const int in_vec_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = nf4_get_pack_factor<32, 4>();
  constexpr int bytes_per_pack = nf4_get_bytes_per_pack();
  constexpr int tn = group_size / pack_factor;
  constexpr int block_size = SIMD_SIZE;
  constexpr int scale_bytes = 4;

  using W_T = uint32_t;
  const device W_T* ws = (const device W_T*)w;

  typedef float U;
  typedef struct {
    W_T wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};
  thread U scale = 0;
  thread U x_local = 0;

  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);
  ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;
  scales +=
      (out_col / group_size + simd_lid * out_vec_size_g) * scale_bytes;
  x += tid.x * in_vec_stride + simd_lid;
  y += tid.x * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  int remaining = in_vec_size % block_size;
  if (remaining == 0) {
    for (int i = 0; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = nf4_dequantize_scale<U, group_size>(scales);
      w_local = *((device vec_w*)ws);
      nf4_qouter<U, tn * pack_factor>(
          (thread uint8_t*)&w_local, x_local, scale, result);
      x += block_size;
      scales += block_size * out_vec_size_g * scale_bytes;
      ws += block_size * out_vec_size_w;
    }
  } else {
    for (int i = block_size; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = nf4_dequantize_scale<U, group_size>(scales);
      w_local = *((device vec_w*)ws);
      nf4_qouter<U, tn * pack_factor>(
          (thread uint8_t*)&w_local, x_local, scale, result);
      x += block_size;
      scales += block_size * out_vec_size_g * scale_bytes;
      ws += block_size * out_vec_size_w;
    }
    if (static_cast<int>(simd_lid) < remaining) {
      x_local = *x;
      scale = nf4_dequantize_scale<U, group_size>(scales);
      w_local = *((device vec_w*)ws);
    } else {
      x_local = 0;
      scale = 0;
    }
    nf4_qouter<U, tn * pack_factor>(
        (thread uint8_t*)&w_local, x_local, scale, result);
  }

#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// QMM kernel (matrix-matrix, transposed weights)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int group_size,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void nf4_qmm_t_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = nf4_get_pack_factor<8, 4>();
  constexpr int bytes_per_pack = nf4_get_bytes_per_pack();
  constexpr int scale_bytes = 4;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = NF4BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g * scale_bytes;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

///////////////////////////////////////////////////////////////////////////////
// QMM kernel (matrix-matrix, non-transposed weights)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int group_size,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void nf4_qmm_n_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = nf4_get_pack_factor<8, 4>();
  constexpr int bytes_per_pack = nf4_get_bytes_per_pack();
  constexpr int scale_bytes = 4;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;
  using loader_w_t = NF4BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size>;

  auto wl = (const device uint8_t*)w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += (y_col / group_size) * scale_bytes;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, N, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Batch offset helpers (reuse from fp_quantized.h via shared include)
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC void nf4_adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device uint8_t*& scales,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr int scale_bytes = 4;
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0] * scale_bytes;
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y * scale_bytes;
  }
  y += tid.z * output_stride;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel entry points
///////////////////////////////////////////////////////////////////////////////

template <typename T, int group_size, int D, bool batched>
[[kernel]] void nf4_qmv_quad(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    nf4_adjust_matrix_offsets(
        x, w, scales, y, out_vec_size * M,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, tid);
  }
  nf4_qmv_quad_impl<T, group_size, D>(
      w, scales, x, y, in_vec_size, out_vec_size, tid, quad_gid, quad_lid);
}

template <typename T, int group_size, bool batched>
[[kernel]] void nf4_qmv_fast(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    nf4_adjust_matrix_offsets(
        x, w, scales, y, out_vec_size * M,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, tid);
  }
  nf4_qmv_fast_impl<T, group_size>(
      w, scales, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, bool batched>
[[kernel]] void nf4_qmv(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    nf4_adjust_matrix_offsets(
        x, w, scales, y, out_vec_size * M,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, tid);
  }
  nf4_qmv_impl<T, group_size>(
      w, scales, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, bool batched>
[[kernel]] void nf4_qvm(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    nf4_adjust_matrix_offsets(
        x, w, scales, y, out_vec_size * M,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, tid);
  }
  nf4_qvm_impl<T, group_size>(
      w, scales, x, y, in_vec_size, out_vec_size, in_vec_size,
      tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, int split_k = 32>
[[kernel]] void nf4_qvm_split_k(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int& final_block_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  nf4_adjust_matrix_offsets(
      x, w, scales, y, out_vec_size * M,
      x_batch_ndims, x_shape, x_strides,
      w_batch_ndims, w_shape, w_strides, s_strides, tid);

  int in_vec_size_adj =
      tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;
  int in_vec_stride = (split_k - 1) * in_vec_size + final_block_size;

  nf4_qvm_impl<T, group_size>(
      w, scales, x, y, in_vec_size_adj, out_vec_size, in_vec_stride,
      tid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const bool aligned_N,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void nf4_qmm_t(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    nf4_adjust_matrix_offsets(
        x, w, scales, y, M * N,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, tid);
  }
  nf4_qmm_t_impl<T, group_size, aligned_N, BM, BK, BN>(
      w, scales, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void nf4_qmm_n(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    nf4_adjust_matrix_offsets(
        x, w, scales, y, M * N,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, tid);
  }
  nf4_qmm_n_impl<T, group_size, BM, BK, BN>(
      w, scales, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void nf4_qmm_t_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& k_partition_size [[buffer(7)]],
    const constant int& split_k_partition_stride [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int pack_factor = nf4_get_pack_factor<8, 4>();
  constexpr int bytes_per_pack = nf4_get_bytes_per_pack();
  constexpr int scale_bytes = 4;
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  const int k_start = tid.z * k_partition_size;
  x += k_start;

  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  scales += (k_start / group_size) * scale_bytes;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  nf4_qmm_t_impl<T, group_size, aligned_N, BM, BK, BN>(
      (const device uint32_t*)wl,
      scales, x, y, Xs, Ws, K, N, M, k_partition_size,
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
// Dequantize standalone kernel
///////////////////////////////////////////////////////////////////////////////

template <typename T, const int group_size>
[[kernel]] void nf4_dequantize(
    const device uint8_t* w [[buffer(0)]],
    const device float* scales_in [[buffer(1)]],
    device T* out [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr int pack_factor = 2; // 4-bit = 2 per byte
  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t oindex = offset * pack_factor;
  size_t gindex = oindex / group_size;

  out += oindex;

  float scale = scales_in[gindex];

  uint val = w[offset];
#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    uint8_t d = (val >> (4 * i)) & 0x0f;
    out[i] = static_cast<T>(scale * nf4_dequantize_value<float>(d));
  }
}
