// Copyright © 2026 Apple Inc.

// Q4_0: 18 bytes/32 weights. [fp16 d][uint8 qs[16]]. w[i] = d * (q4 - 8).

MLX_MTL_CONST int KQ_Q4_0_GROUP = 32;
MLX_MTL_CONST int KQ_Q4_0_BLOCK_BYTES = 18;
MLX_MTL_CONST int KQ_Q4_0_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q4_0_QS_OFFSET = 2;

inline float kq_q4_0_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q4_0_D_OFFSET));
}
inline const device uint8_t* kq_q4_0_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q4_0_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q4_0_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_Q4_0_GROUP;
  const int within = gid % KQ_Q4_0_GROUP;
  const device uint8_t* block_addr = w + block_id * KQ_Q4_0_BLOCK_BYTES;
  const float d = kq_q4_0_d(block_addr);
  const device uint8_t* qs = kq_q4_0_qs_ptr(block_addr);
  const int q4 =
      (within < 16) ? (int(qs[within]) & 0x0F) : (int(qs[within - 16]) >> 4);
  out[gid] = T(d * float(q4 - 8));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q4_0_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q4_0_GROUP, "Q4_0 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_0 kernel requires bits=4");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int row_bytes = in_vec_size * KQ_Q4_0_BLOCK_BYTES / KQ_Q4_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q4_0_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q4_0_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr =
          w + row_idx * row_bytes + ib * KQ_Q4_0_BLOCK_BYTES;
      const U d = U(kq_q4_0_d(block_addr));
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q4_0_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] * U(qi & 0x000F);
        acc[1] += yl[i + 1] * U(qi & 0x0F00);
        acc[2] += yl[i + 8] * U(qi & 0x00F0);
        acc[3] += yl[i + 9] * U(qi & 0xF000);
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3] + sumy * U(-8));
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q4_0_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q4_0_GROUP, "Q4_0 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_0 kernel requires bits=4");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q4_0_BLOCK_BYTES / KQ_Q4_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int nb = in_vec_size / KQ_Q4_0_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q4_0_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr =
          w + row_idx * row_bytes + ib * KQ_Q4_0_BLOCK_BYTES;
      const U d = U(kq_q4_0_d(block_addr));
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q4_0_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] * U(qi & 0x000F);
        acc[1] += yl[i + 1] * U(qi & 0x0F00);
        acc[2] += yl[i + 8] * U(qi & 0x00F0);
        acc[3] += yl[i + 9] * U(qi & 0xF000);
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3] + sumy * U(-8));
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqQ4_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q4_0_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q4_0_BLOCK_BYTES;

  static_assert(
      BCOLS == weights_per_block,
      "Q4_0 loader requires BCOLS == 32 (one block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  MLX_MTL_CONST short half_block = weights_per_block / 2;
  static_assert(n_reads >= 2 && n_reads % 2 == 0, "Q4_0 needs even n_reads.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;

  KqQ4_0BlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int /* col_in_block */ = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const float d = float(*(const device half*)(src + KQ_Q4_0_D_OFFSET));
    const device uint8_t* qs = src + KQ_Q4_0_QS_OFFSET + bj_byte;
#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs[i];
      const int q4_lo = int(b & 0x0F);
      const int q4_hi = int(b >> 4);
      dst[i] = T(d * float(q4_lo - 8));
      dst[half_block + i] = T(d * float(q4_hi - 8));
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[half_block + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    src += tile_stride;
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q4_0_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(
      group_size == KQ_Q4_0_GROUP, "Q4_0 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_0 kernel requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ4_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_q4_0_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q4_0_GROUP, "Q4_0 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_0 kernel requires bits=4");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ4_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;

  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q4_0_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(
      group_size == KQ_Q4_0_GROUP, "Q4_0 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_0 kernel requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ4_0BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q4_0_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_q4_0_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q4_0_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_q4_0_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q4_0_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q4_0_GROUP, "Q4_0 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_0 kernel requires bits=4");
  kq_q4_0_dequantize_impl<T>(w, out, num_weights, gid);
}

// Q4_1: 20 bytes/32 weights. [fp16 d][fp16 m][uint8 qs[16]]. w[i] = d * q4 + m.

MLX_MTL_CONST int KQ_Q4_1_GROUP = 32;
MLX_MTL_CONST int KQ_Q4_1_BLOCK_BYTES = 20;
MLX_MTL_CONST int KQ_Q4_1_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q4_1_M_OFFSET = 2;
MLX_MTL_CONST int KQ_Q4_1_QS_OFFSET = 4;

inline float kq_q4_1_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q4_1_D_OFFSET));
}
inline float kq_q4_1_m(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q4_1_M_OFFSET));
}
inline const device uint8_t* kq_q4_1_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q4_1_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q4_1_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_Q4_1_GROUP;
  const int within = gid % KQ_Q4_1_GROUP;
  const device uint8_t* block_addr = w + block_id * KQ_Q4_1_BLOCK_BYTES;
  const float d = kq_q4_1_d(block_addr);
  const float m = kq_q4_1_m(block_addr);
  const device uint8_t* qs = kq_q4_1_qs_ptr(block_addr);
  const int q4 =
      (within < 16) ? (int(qs[within]) & 0x0F) : (int(qs[within - 16]) >> 4);
  out[gid] = T(d * float(q4) + m);
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q4_1_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q4_1_GROUP, "Q4_1 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_1 kernel requires bits=4");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int row_bytes = in_vec_size * KQ_Q4_1_BLOCK_BYTES / KQ_Q4_1_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q4_1_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q4_1_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr =
          w + row_idx * row_bytes + ib * KQ_Q4_1_BLOCK_BYTES;
      const U d = U(kq_q4_1_d(block_addr));
      const U m = U(kq_q4_1_m(block_addr));
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q4_1_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] * U(qi & 0x000F);
        acc[1] += yl[i + 1] * U(qi & 0x0F00);
        acc[2] += yl[i + 8] * U(qi & 0x00F0);
        acc[3] += yl[i + 9] * U(qi & 0xF000);
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q4_1_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q4_1_GROUP, "Q4_1 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_1 kernel requires bits=4");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q4_1_BLOCK_BYTES / KQ_Q4_1_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int nb = in_vec_size / KQ_Q4_1_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q4_1_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr =
          w + row_idx * row_bytes + ib * KQ_Q4_1_BLOCK_BYTES;
      const U d = U(kq_q4_1_d(block_addr));
      const U m = U(kq_q4_1_m(block_addr));
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q4_1_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] * U(qi & 0x000F);
        acc[1] += yl[i + 1] * U(qi & 0x0F00);
        acc[2] += yl[i + 8] * U(qi & 0x00F0);
        acc[3] += yl[i + 9] * U(qi & 0xF000);
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqQ4_1BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q4_1_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q4_1_BLOCK_BYTES;

  static_assert(
      BCOLS == weights_per_block,
      "Q4_1 loader requires BCOLS == 32 (one block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  MLX_MTL_CONST short half_block = weights_per_block / 2;
  static_assert(n_reads >= 2 && n_reads % 2 == 0, "Q4_1 needs even n_reads.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;

  KqQ4_1BlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int /* col_in_block */ = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const float d = float(*(const device half*)(src + KQ_Q4_1_D_OFFSET));
    const float m = float(*(const device half*)(src + KQ_Q4_1_M_OFFSET));
    const device uint8_t* qs = src + KQ_Q4_1_QS_OFFSET + bj_byte;
    static_assert(
        bytes_per_thread == 4 || bytes_per_thread == 8,
        "Q4_1 ALU vector load supports bytes_per_thread=4 or 8 (uint).");
    uint8_t qs_b[bytes_per_thread];
#pragma unroll
    for (short v = 0; v < bytes_per_thread / 4; v++) {
      const uint qs_v = *reinterpret_cast<const device uint*>(qs + v * 4);
      *reinterpret_cast<thread uint*>(&qs_b[v * 4]) = qs_v;
    }
#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs_b[i];
      const int q4_lo = int(b & 0x0F);
      const int q4_hi = int(b >> 4);
      dst[i] = T(d * float(q4_lo) + m);
      dst[half_block + i] = T(d * float(q4_hi) + m);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[half_block + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    src += tile_stride;
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q4_1_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(
      group_size == KQ_Q4_1_GROUP, "Q4_1 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_1 kernel requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ4_1BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_q4_1_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q4_1_GROUP, "Q4_1 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_1 kernel requires bits=4");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ4_1BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;

  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q4_1_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(
      group_size == KQ_Q4_1_GROUP, "Q4_1 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_1 kernel requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ4_1BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q4_1_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_q4_1_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q4_1_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_q4_1_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q4_1_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q4_1_GROUP, "Q4_1 kernel requires group_size=32");
  static_assert(bits == 4, "Q4_1 kernel requires bits=4");
  kq_q4_1_dequantize_impl<T>(w, out, num_weights, gid);
}

// Q5_0: 22 bytes/32 weights. [fp16 d][uint8 qh[4]][uint8 qs[16]]. w[i] = d *
// (q5 - 16).

MLX_MTL_CONST int KQ_Q5_0_GROUP = 32;
MLX_MTL_CONST int KQ_Q5_0_BLOCK_BYTES = 22;
MLX_MTL_CONST int KQ_Q5_0_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q5_0_QH_OFFSET = 2;
MLX_MTL_CONST int KQ_Q5_0_QS_OFFSET = 6;

inline float kq_q5_0_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q5_0_D_OFFSET));
}
// qh at 22N+2 is not uint32-aligned; assemble from byte loads.
inline uint32_t kq_q5_0_qh(const device uint8_t* block_addr) {
  const device uint8_t* p = block_addr + KQ_Q5_0_QH_OFFSET;
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
      (uint32_t(p[3]) << 24);
}
inline const device uint8_t* kq_q5_0_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q5_0_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q5_0_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_Q5_0_GROUP;
  const int within = gid % KQ_Q5_0_GROUP;
  const device uint8_t* block_addr = w + block_id * KQ_Q5_0_BLOCK_BYTES;
  const float d = kq_q5_0_d(block_addr);
  const uint32_t qh = kq_q5_0_qh(block_addr);
  const device uint8_t* qs = kq_q5_0_qs_ptr(block_addr);
  const uint32_t hi = ((qh >> within) << 4) & 0x10u;
  const uint8_t lo =
      (within < 16) ? (qs[within] & 0x0Fu) : (qs[within - 16] >> 4);
  const int q5 = int(uint32_t(lo) | hi);
  out[gid] = T(d * float(q5 - 16));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q5_0_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_0_GROUP, "Q5_0 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_0 kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int row_bytes = in_vec_size * KQ_Q5_0_BLOCK_BYTES / KQ_Q5_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q5_0_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q5_0_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr =
          w + row_idx * row_bytes + ib * KQ_Q5_0_BLOCK_BYTES;
      const U d = U(kq_q5_0_d(block_addr));
      const uint32_t qh = kq_q5_0_qh(block_addr);
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q5_0_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] *
            U((qi & 0x000F) | (((qh >> (i + 0 + il)) << 4) & 0x00010));
        acc[1] += yl[i + 1] *
            U((qi & 0x0F00) | (((qh >> (i + 1 + il)) << 12) & 0x01000));
        acc[2] += yl[i + 8] *
            U((qi & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8) & 0x00100));
        acc[3] += yl[i + 9] *
            U((qi & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3] + sumy * U(-16));
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q5_0_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_0_GROUP, "Q5_0 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_0 kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q5_0_BLOCK_BYTES / KQ_Q5_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int nb = in_vec_size / KQ_Q5_0_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q5_0_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr =
          w + row_idx * row_bytes + ib * KQ_Q5_0_BLOCK_BYTES;
      const U d = U(kq_q5_0_d(block_addr));
      const uint32_t qh = kq_q5_0_qh(block_addr);
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q5_0_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] *
            U((qi & 0x000F) | (((qh >> (i + 0 + il)) << 4) & 0x00010));
        acc[1] += yl[i + 1] *
            U((qi & 0x0F00) | (((qh >> (i + 1 + il)) << 12) & 0x01000));
        acc[2] += yl[i + 8] *
            U((qi & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8) & 0x00100));
        acc[3] += yl[i + 9] *
            U((qi & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3] + sumy * U(-16));
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqQ5_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q5_0_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q5_0_BLOCK_BYTES;

  static_assert(
      BCOLS == weights_per_block,
      "Q5_0 loader requires BCOLS == 32 (one block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  MLX_MTL_CONST short half_block = weights_per_block / 2;
  static_assert(n_reads >= 2 && n_reads % 2 == 0, "Q5_0 needs even n_reads.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;

  KqQ5_0BlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int /* col_in_block */ = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const float d = float(*(const device half*)(src + KQ_Q5_0_D_OFFSET));
    const device uint8_t* qh_p = src + KQ_Q5_0_QH_OFFSET;
    const uint32_t qh = uint32_t(qh_p[0]) | (uint32_t(qh_p[1]) << 8) |
        (uint32_t(qh_p[2]) << 16) | (uint32_t(qh_p[3]) << 24);
    const device uint8_t* qs = src + KQ_Q5_0_QS_OFFSET + bj_byte;
#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs[i];
      const int j_lo = bj_byte + i;
      const int j_hi = bj_byte + half_block + i;
      const uint32_t hi_lo = ((qh >> j_lo) << 4) & 0x10u;
      const uint32_t hi_hi = ((qh >> j_hi) << 4) & 0x10u;
      const int q5_lo = int(uint32_t(b & 0x0F) | hi_lo);
      const int q5_hi = int(uint32_t(b >> 4) | hi_hi);
      dst[i] = T(d * float(q5_lo - 16));
      dst[half_block + i] = T(d * float(q5_hi - 16));
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[half_block + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    src += tile_stride;
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q5_0_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(
      group_size == KQ_Q5_0_GROUP, "Q5_0 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_0 kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_q5_0_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q5_0_GROUP, "Q5_0 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_0 kernel requires bits=5");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;

  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_0_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(
      group_size == KQ_Q5_0_GROUP, "Q5_0 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_0 kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ5_0BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_0_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_q5_0_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_0_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
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
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_q5_0_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_0_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q5_0_GROUP, "Q5_0 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_0 kernel requires bits=5");
  kq_q5_0_dequantize_impl<T>(w, out, num_weights, gid);
}
