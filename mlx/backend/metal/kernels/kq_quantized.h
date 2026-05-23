// Copyright © 2026 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

struct kq_empty {};

template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void kq_qmm_t_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const int K_eff,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be >= SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be a multiple of SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      /*transpose_a=*/false,
      /*transpose_b=*/true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = w;

  x += y_row * static_cast<int64_t>(K);
  wl += static_cast<int64_t>(y_col) * K_w;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  LoaderW loader_w(wl, K, Ws, simd_gid, simd_lid);
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

template <
    typename T,
    typename LoaderW,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void kq_qmm_n_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be >= SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be a multiple of SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      /*transpose_a=*/false,
      /*transpose_b=*/false,
      BK_padded,
      BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;

  auto wl = w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += (y_col / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  LoaderW loader_w(
      wl, N, Ws, simd_gid, simd_lid, y_col % LoaderW::weights_per_block);
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

template <typename T>
METAL_FUNC void kq_adjust_matrix_offsets(
    const device T*& x,
    const device uint8_t*& w,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
  } else {
    w += elem_to_loc(w_idx, w_shape, w_strides, w_batch_ndims);
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void kq_adjust_matrix_offsets(
    const device T*& x,
    const device uint8_t*& w,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
  } else {
    w += elem_to_loc(w_idx, w_shape, w_strides, w_batch_ndims);
  }
  y += tid.z * output_stride;
}

// Q8_0: 34 bytes/32 weights. [fp16 d][int8 q[32]]. w[i] = d * q[i].

MLX_MTL_CONST int KQ_Q8_0_GROUP = 32;
MLX_MTL_CONST int KQ_Q8_0_BLOCK_BYTES = 34;
MLX_MTL_CONST int KQ_Q8_0_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q8_0_Q_OFFSET = 2;

inline float kq_q8_0_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q8_0_D_OFFSET));
}

inline const device int8_t* kq_q8_0_q_ptr(const device uint8_t* block_addr) {
  return (const device int8_t*)(block_addr + KQ_Q8_0_Q_OFFSET);
}

template <typename T>
METAL_FUNC void kq_q8_0_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_Q8_0_GROUP;
  const int within = gid % KQ_Q8_0_GROUP;
  const device uint8_t* block_addr = w + block_id * KQ_Q8_0_BLOCK_BYTES;
  const float d = kq_q8_0_d(block_addr);
  const int8_t q = kq_q8_0_q_ptr(block_addr)[within];
  out[gid] = T(d * float(q));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q8_0_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int lane_k_offset = simd_lid * values_per_thread;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x + k + lane_k_offset, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* row_base = w + row_idx * row_bytes;

      const int k_global = k + lane_k_offset;
      const int block_id = k_global / KQ_Q8_0_GROUP;
      const int within = k_global - block_id * KQ_Q8_0_GROUP;
      const device uint8_t* block_addr =
          row_base + block_id * KQ_Q8_0_BLOCK_BYTES;

      const U d = U(kq_q8_0_d(block_addr));
      const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;

      U partial = 0;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        partial += x_thread[i] * U(q_ptr[i]);
      }
      result[row] += d * partial;
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
METAL_FUNC void kq_q8_0_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int lane_k_offset = simd_lid * values_per_thread;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int k_remaining = in_vec_size - k - lane_k_offset;
    if (k_remaining >= values_per_thread) {
      load_vector<T, U, values_per_thread>(x + k + lane_k_offset, x_thread);
    } else if (k_remaining > 0) {
      load_vector_safe<T, U, values_per_thread>(
          x + k + lane_k_offset, x_thread, k_remaining);
    } else {
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        x_thread[i] = 0;
      }
    }

    const int n_inner = k_remaining >= values_per_thread
        ? values_per_thread
        : (k_remaining > 0 ? k_remaining : 0);

    if (n_inner == 0) {
      continue;
    }

    const int k_global = k + lane_k_offset;
    const int block_id = k_global / KQ_Q8_0_GROUP;
    const int within = k_global - block_id * KQ_Q8_0_GROUP;

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* row_base = w + row_idx * row_bytes;
      const device uint8_t* block_addr =
          row_base + block_id * KQ_Q8_0_BLOCK_BYTES;

      const U d = U(kq_q8_0_d(block_addr));
      const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;

      U partial = 0;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        if (i < n_inner) {
          partial += x_thread[i] * U(q_ptr[i]);
        }
      }
      result[row] += d * partial;
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
struct KqQ8_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q8_0_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q8_0_BLOCK_BYTES;

  static_assert(
      BCOLS == weights_per_block,
      "Q8_0 loader requires BCOLS == 32 (one block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqQ8_0BlockLoader(
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
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const float d = float(*(const device half*)src);
    const device int8_t* q =
        (const device int8_t*)(src + KQ_Q8_0_Q_OFFSET + bj);
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      dst[i] = T(d * float(q[i]));
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    const float d = float(*(const device half*)src);
    const device int8_t* q =
        (const device int8_t*)(src + KQ_Q8_0_Q_OFFSET + bj);
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      dst[i] = T(d * float(q[i]));
    }
  }

  void next() {
    src += tile_stride;
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q8_0_qmm_t(
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
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ8_0BlockLoader<
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
[[kernel]] void kq_q8_0_qmm_t_splitk(
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
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ8_0BlockLoader<
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
[[kernel]] void kq_q8_0_qmm_n(
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
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ8_0BlockLoader<
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
[[kernel]] void kq_q8_0_qmv_fast(
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
  kq_q8_0_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_qmv(
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
  kq_q8_0_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q8_0_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  kq_q8_0_dequantize_impl<T>(w, out, num_weights, gid);
}

#include "mlx/backend/metal/kernels/kq_quantized_legacy.h"

// Q5_1: 24 bytes/32 weights. [fp16 d][fp16 m][uint32 qh][uint8 qs[16]].
// q5 = (low4 | high_bit<<4); w[i] = d * q5[i] + m.

MLX_MTL_CONST int KQ_Q5_1_GROUP = 32;
MLX_MTL_CONST int KQ_Q5_1_BLOCK_BYTES = 24;
MLX_MTL_CONST int KQ_Q5_1_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q5_1_M_OFFSET = 2;
MLX_MTL_CONST int KQ_Q5_1_QH_OFFSET = 4;
MLX_MTL_CONST int KQ_Q5_1_QS_OFFSET = 8;

inline float kq_q5_1_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q5_1_D_OFFSET));
}
inline float kq_q5_1_m(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q5_1_M_OFFSET));
}
inline uint32_t kq_q5_1_qh(const device uint8_t* block_addr) {
  return *(const device uint32_t*)(block_addr + KQ_Q5_1_QH_OFFSET);
}
inline const device uint8_t* kq_q5_1_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q5_1_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q5_1_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_Q5_1_GROUP;
  const int within = gid % KQ_Q5_1_GROUP;
  const device uint8_t* block_addr = w + block_id * KQ_Q5_1_BLOCK_BYTES;
  const float d = kq_q5_1_d(block_addr);
  const float m = kq_q5_1_m(block_addr);
  const uint32_t qh = kq_q5_1_qh(block_addr);
  const device uint8_t* qs = kq_q5_1_qs_ptr(block_addr);
  const uint32_t hi = ((qh >> within) << 4) & 0x10u;
  const uint8_t lo =
      (within < 16) ? (qs[within] & 0x0Fu) : (qs[within - 16] >> 4);
  const float q5 = float(uint32_t(lo) | hi);
  out[gid] = T(d * q5 + m);
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q5_1_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int row_bytes = in_vec_size * KQ_Q5_1_BLOCK_BYTES / KQ_Q5_1_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q5_1_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q5_1_GROUP + il;
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
          w + row_idx * row_bytes + ib * KQ_Q5_1_BLOCK_BYTES;
      const U d = U(kq_q5_1_d(block_addr));
      const U m = U(kq_q5_1_m(block_addr));
      const uint32_t qh = kq_q5_1_qh(block_addr);
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q5_1_qs_ptr(block_addr)) +
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
METAL_FUNC void kq_q5_1_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q5_1_BLOCK_BYTES / KQ_Q5_1_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int nb = in_vec_size / KQ_Q5_1_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q5_1_GROUP + il;
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
          w + row_idx * row_bytes + ib * KQ_Q5_1_BLOCK_BYTES;
      const U d = U(kq_q5_1_d(block_addr));
      const U m = U(kq_q5_1_m(block_addr));
      const uint32_t qh = kq_q5_1_qh(block_addr);
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q5_1_qs_ptr(block_addr)) +
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
struct KqQ5_1BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q5_1_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q5_1_BLOCK_BYTES;

  static_assert(
      BCOLS == weights_per_block,
      "Q5_1 loader requires BCOLS == 32 (one block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  MLX_MTL_CONST short half_block = weights_per_block / 2;
  static_assert(n_reads >= 2 && n_reads % 2 == 0, "Q5_1 needs even n_reads.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;

  KqQ5_1BlockLoader(
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
    const float d = float(*(const device half*)(src + KQ_Q5_1_D_OFFSET));
    const float m = float(*(const device half*)(src + KQ_Q5_1_M_OFFSET));
    const uint32_t qh = *(const device uint32_t*)(src + KQ_Q5_1_QH_OFFSET);
    const device uint8_t* qs = src + KQ_Q5_1_QS_OFFSET + bj_byte;
    static_assert(
        bytes_per_thread == 4 || bytes_per_thread == 8,
        "Q5_1 ALU vector load supports bytes_per_thread=4 or 8 (uint).");
    uint8_t qs_b[bytes_per_thread];
#pragma unroll
    for (short v = 0; v < bytes_per_thread / 4; v++) {
      const uint qs_v = *reinterpret_cast<const device uint*>(qs + v * 4);
      *reinterpret_cast<thread uint*>(&qs_b[v * 4]) = qs_v;
    }
#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs_b[i];
      const int j_lo = bj_byte + i;
      const int j_hi = bj_byte + half_block + i;
      const uint32_t hi_lo = ((qh >> j_lo) << 4) & 0x10u;
      const uint32_t hi_hi = ((qh >> j_hi) << 4) & 0x10u;
      const float q5_lo = float(uint32_t(b & 0x0F) | hi_lo);
      const float q5_hi = float(uint32_t(b >> 4) | hi_hi);
      dst[i] = T(d * q5_lo + m);
      dst[half_block + i] = T(d * q5_hi + m);
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
[[kernel]] void kq_q5_1_qmm_t(
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
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_1BlockLoader<
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
[[kernel]] void kq_q5_1_qmm_t_splitk(
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
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_1BlockLoader<
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
[[kernel]] void kq_q5_1_qmm_n(
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
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ5_1BlockLoader<
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
[[kernel]] void kq_q5_1_qmv_fast(
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
  kq_q5_1_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_1_qmv(
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
  kq_q5_1_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_1_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  kq_q5_1_dequantize_impl<T>(w, out, num_weights, gid);
}

inline void kq_get_scale_min_k4(
    int j,
    const device uint8_t* q,
    thread uint8_t& d_out,
    thread uint8_t& m_out) {
  const int j_lo = j & 3;
  const bool j_high = (j & 4) != 0;
  const uint8_t a = q[j_lo];
  const uint8_t b = q[j_lo + 4];
  const uint8_t c = q[j_lo + 8];
  const uint8_t d_low = a & 0x3F;
  const uint8_t m_low = b & 0x3F;
  const uint8_t d_high = (c & 0x0F) | ((a >> 6) << 4);
  const uint8_t m_high = (c >> 4) | ((b >> 6) << 4);
  d_out = j_high ? d_high : d_low;
  m_out = j_high ? m_high : m_low;
}

// Q4_K: 144 bytes/256 weights. [fp16 d][fp16 dmin][scales[12]][qs[128]].
// w[i] = d * sub_scale * q4 - dmin * sub_min. Nibble-packed, low=even sb.

MLX_MTL_CONST int KQ_Q4_K_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_Q4_K_BLOCK_BYTES = 144;
MLX_MTL_CONST int KQ_Q4_K_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q4_K_DMIN_OFFSET = 2;
MLX_MTL_CONST int KQ_Q4_K_SCALES_OFFSET = 4;
MLX_MTL_CONST int KQ_Q4_K_QS_OFFSET = 16;

inline float kq_q4_k_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q4_K_D_OFFSET));
}
inline float kq_q4_k_dmin(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q4_K_DMIN_OFFSET));
}
inline const device uint8_t* kq_q4_k_scales12_ptr(
    const device uint8_t* block_addr) {
  return block_addr + KQ_Q4_K_SCALES_OFFSET;
}
inline const device uint8_t* kq_q4_k_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q4_K_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q4_k_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_Q4_K_SUPERBLOCK;
  const int within_sb_total = gid - sb_id * KQ_Q4_K_SUPERBLOCK;
  const int sub_block = within_sb_total / 32;
  const int within_sb = within_sb_total - sub_block * 32;
  const int pair = sub_block / 2;
  const bool is_high = (sub_block & 1) != 0;
  const int qs_byte_idx = pair * 32 + within_sb;

  const device uint8_t* sb_addr = w + sb_id * KQ_Q4_K_BLOCK_BYTES;
  const float d = kq_q4_k_d(sb_addr);
  const float dmin = kq_q4_k_dmin(sb_addr);
  uint8_t sc6, mn6;
  kq_get_scale_min_k4(sub_block, kq_q4_k_scales12_ptr(sb_addr), sc6, mn6);

  const uint8_t byte = kq_q4_k_qs_ptr(sb_addr)[qs_byte_idx];
  const uint8_t q4 = is_high ? (byte >> 4) : (byte & 0x0F);
  out[gid] = T(d * float(sc6) * float(q4) - dmin * float(mn6));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q4_k_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q4_K_SUPERBLOCK, "Q4_K kernel requires group_size=256");
  static_assert(bits == 4, "Q4_K kernel requires bits=4");

  constexpr int num_simdgroups = 2;
  // 2 (vs 4 for flat codecs), super-block scale unpacking needs more registers.
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;

  typedef float U;
  thread U yl[16];
  thread U yh[16];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 8;
  const int it = simd_lid % 8;
  const int iq = it / 4;
  const int ir = it % 4;

  const int row_bytes = in_vec_size * KQ_Q4_K_BLOCK_BYTES / KQ_Q4_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q4_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q4_K_SUPERBLOCK + 64 * iq + 8 * ir;
    U sumy[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      yl[i + 0] = U(x[x_base + i + 0]);
      sumy[0] += yl[i + 0];
      yl[i + 8] = U(x[x_base + i + 32]);
      sumy[1] += yl[i + 8];
      yh[i + 0] = U(x[x_base + i + 128]);
      sumy[2] += yh[i + 0];
      yh[i + 8] = U(x[x_base + i + 160]);
      sumy[3] += yh[i + 8];
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q4_K_BLOCK_BYTES;

      const device uint16_t* sc16_src =
          reinterpret_cast<const device uint16_t*>(
              kq_q4_k_scales12_ptr(sb_addr)) +
          iq;
      uint16_t sc16[4];
      sc16[0] = sc16_src[0] & kmask1;
      sc16[1] = sc16_src[2] & kmask1;
      sc16[2] = ((sc16_src[4] >> 0) & kmask2) | ((sc16_src[0] & kmask3) >> 2);
      sc16[3] = ((sc16_src[4] >> 4) & kmask2) | ((sc16_src[2] & kmask3) >> 2);
      thread const uint8_t* sc8 = reinterpret_cast<thread const uint8_t*>(sc16);

      const device uint16_t* q1 =
          reinterpret_cast<const device uint16_t*>(kq_q4_k_qs_ptr(sb_addr)) +
          16 * iq + 4 * ir;
      const device uint16_t* q2 = q1 + 32;

      U acc1[4] = {U(0), U(0), U(0), U(0)};
      U acc2[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint16_t q1i = q1[i];
        const uint16_t q2i = q2[i];
        acc1[0] += yl[2 * i + 0] * U(q1i & 0x000F);
        acc1[1] += yl[2 * i + 1] * U(q1i & 0x0F00);
        acc1[2] += yl[2 * i + 8] * U(q1i & 0x00F0);
        acc1[3] += yl[2 * i + 9] * U(q1i & 0xF000);
        acc2[0] += yh[2 * i + 0] * U(q2i & 0x000F);
        acc2[1] += yh[2 * i + 1] * U(q2i & 0x0F00);
        acc2[2] += yh[2 * i + 8] * U(q2i & 0x00F0);
        acc2[3] += yh[2 * i + 9] * U(q2i & 0xF000);
      }

      const U d = U(kq_q4_k_d(sb_addr));
      const U dmin = U(kq_q4_k_dmin(sb_addr));
      result[row] += d *
              ((acc1[0] + acc1[1] * (U(1) / U(256))) * U(sc8[0]) +
               (acc1[2] + acc1[3] * (U(1) / U(256))) * U(sc8[1]) *
                   (U(1) / U(16)) +
               (acc2[0] + acc2[1] * (U(1) / U(256))) * U(sc8[4]) +
               (acc2[2] + acc2[3] * (U(1) / U(256))) * U(sc8[5]) *
                   (U(1) / U(16))) -
          dmin *
              (sumy[0] * U(sc8[2]) + sumy[1] * U(sc8[3]) + sumy[2] * U(sc8[6]) +
               sumy[3] * U(sc8[7]));
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
METAL_FUNC void kq_q4_k_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q4_K_SUPERBLOCK, "Q4_K kernel requires group_size=256");
  static_assert(bits == 4, "Q4_K kernel requires bits=4");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q4_K_BLOCK_BYTES / KQ_Q4_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int lane_k_offset = simd_lid * values_per_thread;
  const int sub_block = simd_lid / 4;
  const int within_sb = (simd_lid % 4) * values_per_thread;
  const int pair = sub_block / 2;
  const bool is_high = (sub_block & 1) != 0;
  const int qs_byte_idx = pair * 32 + within_sb;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x + k + lane_k_offset, x_thread);

    U partial_x = 0;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      partial_x += x_thread[i];
    }

    const int sb_id = k / KQ_Q4_K_SUPERBLOCK;
    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + sb_id * KQ_Q4_K_BLOCK_BYTES;

      const U d = U(kq_q4_k_d(sb_addr));
      const U dmin = U(kq_q4_k_dmin(sb_addr));
      uint8_t sc6, mn6;
      kq_get_scale_min_k4(sub_block, kq_q4_k_scales12_ptr(sb_addr), sc6, mn6);
      const U eff_scale = d * U(sc6);
      const U eff_min = dmin * U(mn6);

      const device uint8_t* qs = kq_q4_k_qs_ptr(sb_addr) + qs_byte_idx;
      U partial_q = 0;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        const uint8_t q4 = is_high ? (qs[i] >> 4) : (qs[i] & 0x0F);
        partial_q += x_thread[i] * U(q4);
      }
      result[row] += eff_scale * partial_q - eff_min * partial_x;
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
struct KqQ4_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q4_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q4_K_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;

  static_assert(
      BCOLS == sub_block_size,
      "Q4_K loader requires BCOLS == 32 (one sub-block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_sub_block_idx;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short sub_block_idx;
  struct Cache {
    T vals[n_reads];
  };
  metal::conditional_t<reduction_dim == 1, Cache, kq_empty> cached;

  KqQ4_KBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      if (sub_block_idx & 1) {
#pragma unroll
        for (short i = 0; i < n_reads; i++) {
          dst[i] = cached.vals[i];
        }
        return;
      }
    }

    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;

    const float d = float(*(const device half*)(src + KQ_Q4_K_D_OFFSET));
    const float dmin = float(*(const device half*)(src + KQ_Q4_K_DMIN_OFFSET));
    const device uint8_t* scales12 = src + KQ_Q4_K_SCALES_OFFSET;

    uint8_t sc6, mn6;
    kq_get_scale_min_k4(sb, scales12, sc6, mn6);
    const float eff_scale = d * float(sc6);
    const float eff_min = dmin * float(mn6);

    const short pair = sb / 2;
    const device uint8_t* qs = src + KQ_Q4_K_QS_OFFSET + pair * 32 + bj;

    static_assert(
        n_reads == 8 || n_reads == 16,
        "Q4_K ALU vector load supports n_reads=8 (uint2) or 16 (uint4).");
    uint8_t qs_b[n_reads];
    if constexpr (n_reads == 8) {
      const uint2 qs_v = *reinterpret_cast<const device uint2*>(qs);
      *reinterpret_cast<thread uint2*>(&qs_b[0]) = qs_v;
    } else {
      const uint4 qs_v = *reinterpret_cast<const device uint4*>(qs);
      *reinterpret_cast<thread uint4*>(&qs_b[0]) = qs_v;
    }

    if constexpr (reduction_dim == 1) {
      uint8_t sc6_hi, mn6_hi;
      kq_get_scale_min_k4(sb + 1, scales12, sc6_hi, mn6_hi);
      const float eff_scale_hi = d * float(sc6_hi);
      const float eff_min_hi = dmin * float(mn6_hi);

#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t b = qs_b[i];
        const uint8_t q4_lo = b & 0x0F;
        const uint8_t q4_hi = b >> 4;
        dst[i] = T(eff_scale * float(q4_lo) - eff_min);
        cached.vals[i] = T(eff_scale_hi * float(q4_hi) - eff_min_hi);
      }
    } else {
      const bool is_high = (sb & 1) != 0;
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t q4 = is_high ? (qs_b[i] >> 4) : (qs_b[i] & 0x0F);
        dst[i] = T(eff_scale * float(q4) - eff_min);
      }
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      sub_block_idx++;
      if (sub_block_idx == sub_blocks_per_block) {
        sub_block_idx = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q4_k_qmm_t(
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
      group_size == KQ_Q4_K_SUPERBLOCK, "Q4_K kernel requires group_size=256");
  static_assert(bits == 4, "Q4_K kernel requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ4_KBlockLoader<
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
[[kernel]] void kq_q4_k_qmm_t_splitk(
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
      group_size == KQ_Q4_K_SUPERBLOCK, "Q4_K kernel requires group_size=256");
  static_assert(bits == 4, "Q4_K kernel requires bits=4");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ4_KBlockLoader<
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
[[kernel]] void kq_q4_k_qmm_n(
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
      group_size == KQ_Q4_K_SUPERBLOCK, "Q4_K kernel requires group_size=256");
  static_assert(bits == 4, "Q4_K kernel requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ4_KBlockLoader<
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
[[kernel]] void kq_q4_k_qmv_fast(
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
  kq_q4_k_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q4_k_qmv(
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
  kq_q4_k_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q4_k_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q4_K_SUPERBLOCK, "Q4_K kernel requires group_size=256");
  static_assert(bits == 4, "Q4_K kernel requires bits=4");
  kq_q4_k_dequantize_impl<T>(w, out, num_weights, gid);
}

// Q5_K: 176 bytes/256 weights. [fp16 d][fp16
// dmin][scales[12]][qh[32]][qs[128]]. q5 = q4 | (high_bit<<4); w[i] = d *
// sub_scale * q5 - dmin * sub_min.

MLX_MTL_CONST int KQ_Q5_K_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_Q5_K_BLOCK_BYTES = 176;
MLX_MTL_CONST int KQ_Q5_K_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q5_K_DMIN_OFFSET = 2;
MLX_MTL_CONST int KQ_Q5_K_SCALES_OFFSET = 4;
MLX_MTL_CONST int KQ_Q5_K_QH_OFFSET = 16;
MLX_MTL_CONST int KQ_Q5_K_QS_OFFSET = 48;

inline float kq_q5_k_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q5_K_D_OFFSET));
}
inline float kq_q5_k_dmin(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q5_K_DMIN_OFFSET));
}
inline const device uint8_t* kq_q5_k_scales12_ptr(
    const device uint8_t* block_addr) {
  return block_addr + KQ_Q5_K_SCALES_OFFSET;
}
inline const device uint8_t* kq_q5_k_qh_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q5_K_QH_OFFSET;
}
inline const device uint8_t* kq_q5_k_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q5_K_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q5_k_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_Q5_K_SUPERBLOCK;
  const int within_sb_total = gid - sb_id * KQ_Q5_K_SUPERBLOCK;
  const int sub_block = within_sb_total / 32;
  const int within_sb = within_sb_total - sub_block * 32;
  const int pair = sub_block / 2;
  const bool is_high = (sub_block & 1) != 0;
  const int qs_byte_idx = pair * 32 + within_sb;

  const device uint8_t* sb_addr = w + sb_id * KQ_Q5_K_BLOCK_BYTES;
  const float d = kq_q5_k_d(sb_addr);
  const float dmin = kq_q5_k_dmin(sb_addr);
  uint8_t sc6, mn6;
  kq_get_scale_min_k4(sub_block, kq_q5_k_scales12_ptr(sb_addr), sc6, mn6);

  const uint8_t qs_byte = kq_q5_k_qs_ptr(sb_addr)[qs_byte_idx];
  const uint8_t q4 = is_high ? (qs_byte >> 4) : (qs_byte & 0x0F);
  const uint8_t qh_byte = kq_q5_k_qh_ptr(sb_addr)[within_sb];
  const uint8_t high_bit = (qh_byte >> sub_block) & 1u;
  const uint8_t q5 = q4 | (high_bit << 4);
  out[gid] = T(d * float(sc6) * float(q5) - dmin * float(mn6));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q5_k_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_K_SUPERBLOCK, "Q5_K kernel requires group_size=256");
  static_assert(bits == 5, "Q5_K kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;

  typedef float U;
  thread U yl[16];
  thread U yh[16];
  thread U result[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 4;
  const int ix = simd_lid % 4;
  const int iq = tid_lane / 4;
  const int ir = tid_lane % 4;

  const uint8_t hm1 = uint8_t(1u << (2 * iq));
  const uint8_t hm2 = uint8_t(hm1 << 1);
  const uint8_t hm3 = uint8_t(hm1 << 4);
  const uint8_t hm4 = uint8_t(hm2 << 4);

  const int row_bytes = in_vec_size * KQ_Q5_K_BLOCK_BYTES / KQ_Q5_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q5_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q5_K_SUPERBLOCK + 64 * iq + 8 * ir;
    U sumy[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      yl[i + 0] = U(x[x_base + i + 0]);
      sumy[0] += yl[i + 0];
      yl[i + 8] = U(x[x_base + i + 32]);
      sumy[1] += yl[i + 8];
      yh[i + 0] = U(x[x_base + i + 128]);
      sumy[2] += yh[i + 0];
      yh[i + 8] = U(x[x_base + i + 160]);
      sumy[3] += yh[i + 8];
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q5_K_BLOCK_BYTES;

      const device uint16_t* sc16_src =
          reinterpret_cast<const device uint16_t*>(
              kq_q5_k_scales12_ptr(sb_addr)) +
          iq;
      uint16_t sc16[4];
      sc16[0] = sc16_src[0] & kmask1;
      sc16[1] = sc16_src[2] & kmask1;
      sc16[2] = ((sc16_src[4] >> 0) & kmask2) | ((sc16_src[0] & kmask3) >> 2);
      sc16[3] = ((sc16_src[4] >> 4) & kmask2) | ((sc16_src[2] & kmask3) >> 2);
      thread const uint8_t* sc8 = reinterpret_cast<thread const uint8_t*>(sc16);

      const device uint16_t* q1 =
          reinterpret_cast<const device uint16_t*>(kq_q5_k_qs_ptr(sb_addr)) +
          16 * iq + 4 * ir;
      const device uint16_t* q2 = q1 + 32;
      const device uint8_t* qh = kq_q5_k_qh_ptr(sb_addr) + 8 * ir;

      U acc1[4] = {U(0), U(0), U(0), U(0)};
      U acc2[4] = {U(0), U(0), U(0), U(0)};
      U accH[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint16_t q1i = q1[i];
        const uint16_t q2i = q2[i];
        const uint8_t h0 = qh[2 * i + 0];
        const uint8_t h1 = qh[2 * i + 1];
        acc1[0] += yl[2 * i + 0] * U(q1i & 0x000F);
        acc1[1] += yl[2 * i + 1] * U(q1i & 0x0F00);
        acc1[2] += yl[2 * i + 8] * U(q1i & 0x00F0);
        acc1[3] += yl[2 * i + 9] * U(q1i & 0xF000);
        acc2[0] += yh[2 * i + 0] * U(q2i & 0x000F);
        acc2[1] += yh[2 * i + 1] * U(q2i & 0x0F00);
        acc2[2] += yh[2 * i + 8] * U(q2i & 0x00F0);
        acc2[3] += yh[2 * i + 9] * U(q2i & 0xF000);
        accH[0] += ((h0 & hm1) ? yl[2 * i + 0] : U(0)) +
            ((h1 & hm1) ? yl[2 * i + 1] : U(0));
        accH[1] += ((h0 & hm2) ? yl[2 * i + 8] : U(0)) +
            ((h1 & hm2) ? yl[2 * i + 9] : U(0));
        accH[2] += ((h0 & hm3) ? yh[2 * i + 0] : U(0)) +
            ((h1 & hm3) ? yh[2 * i + 1] : U(0));
        accH[3] += ((h0 & hm4) ? yh[2 * i + 8] : U(0)) +
            ((h1 & hm4) ? yh[2 * i + 9] : U(0));
      }

      const U d = U(kq_q5_k_d(sb_addr));
      const U dmin = U(kq_q5_k_dmin(sb_addr));
      result[row] += d *
              (U(sc8[0]) *
                   ((acc1[0] + acc1[1] * (U(1) / U(256))) + U(16) * accH[0]) +
               U(sc8[1]) *
                   ((acc1[2] + acc1[3] * (U(1) / U(256))) * (U(1) / U(16)) +
                    U(16) * accH[1]) +
               U(sc8[4]) *
                   ((acc2[0] + acc2[1] * (U(1) / U(256))) + U(16) * accH[2]) +
               U(sc8[5]) *
                   ((acc2[2] + acc2[3] * (U(1) / U(256))) * (U(1) / U(16)) +
                    U(16) * accH[3])) -
          dmin *
              (sumy[0] * U(sc8[2]) + sumy[1] * U(sc8[3]) + sumy[2] * U(sc8[6]) +
               sumy[3] * U(sc8[7]));
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
METAL_FUNC void kq_q5_k_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_K_SUPERBLOCK, "Q5_K kernel requires group_size=256");
  static_assert(bits == 5, "Q5_K kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;
  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;

  typedef float U;
  thread U yl[16];
  thread U yh[16];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q5_K_BLOCK_BYTES / KQ_Q5_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int tid_lane = simd_lid / 4;
  const int ix = simd_lid % 4;
  const int iq = tid_lane / 4;
  const int ir = tid_lane % 4;

  const uint8_t hm1 = uint8_t(1u << (2 * iq));
  const uint8_t hm2 = uint8_t(hm1 << 1);
  const uint8_t hm3 = uint8_t(hm1 << 4);
  const uint8_t hm4 = uint8_t(hm2 << 4);

  const int nb = in_vec_size / KQ_Q5_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q5_K_SUPERBLOCK + 64 * iq + 8 * ir;
    U sumy[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      yl[i + 0] = U(x[x_base + i + 0]);
      sumy[0] += yl[i + 0];
      yl[i + 8] = U(x[x_base + i + 32]);
      sumy[1] += yl[i + 8];
      yh[i + 0] = U(x[x_base + i + 128]);
      sumy[2] += yh[i + 0];
      yh[i + 8] = U(x[x_base + i + 160]);
      sumy[3] += yh[i + 8];
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q5_K_BLOCK_BYTES;

      const device uint16_t* sc16_src =
          reinterpret_cast<const device uint16_t*>(
              kq_q5_k_scales12_ptr(sb_addr)) +
          iq;
      uint16_t sc16[4];
      sc16[0] = sc16_src[0] & kmask1;
      sc16[1] = sc16_src[2] & kmask1;
      sc16[2] = ((sc16_src[4] >> 0) & kmask2) | ((sc16_src[0] & kmask3) >> 2);
      sc16[3] = ((sc16_src[4] >> 4) & kmask2) | ((sc16_src[2] & kmask3) >> 2);
      thread const uint8_t* sc8 = reinterpret_cast<thread const uint8_t*>(sc16);

      const device uint16_t* q1 =
          reinterpret_cast<const device uint16_t*>(kq_q5_k_qs_ptr(sb_addr)) +
          16 * iq + 4 * ir;
      const device uint16_t* q2 = q1 + 32;
      const device uint8_t* qh = kq_q5_k_qh_ptr(sb_addr) + 8 * ir;

      U acc1[4] = {U(0), U(0), U(0), U(0)};
      U acc2[4] = {U(0), U(0), U(0), U(0)};
      U accH[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint16_t q1i = q1[i];
        const uint16_t q2i = q2[i];
        const uint8_t h0 = qh[2 * i + 0];
        const uint8_t h1 = qh[2 * i + 1];
        acc1[0] += yl[2 * i + 0] * U(q1i & 0x000F);
        acc1[1] += yl[2 * i + 1] * U(q1i & 0x0F00);
        acc1[2] += yl[2 * i + 8] * U(q1i & 0x00F0);
        acc1[3] += yl[2 * i + 9] * U(q1i & 0xF000);
        acc2[0] += yh[2 * i + 0] * U(q2i & 0x000F);
        acc2[1] += yh[2 * i + 1] * U(q2i & 0x0F00);
        acc2[2] += yh[2 * i + 8] * U(q2i & 0x00F0);
        acc2[3] += yh[2 * i + 9] * U(q2i & 0xF000);
        accH[0] += ((h0 & hm1) ? yl[2 * i + 0] : U(0)) +
            ((h1 & hm1) ? yl[2 * i + 1] : U(0));
        accH[1] += ((h0 & hm2) ? yl[2 * i + 8] : U(0)) +
            ((h1 & hm2) ? yl[2 * i + 9] : U(0));
        accH[2] += ((h0 & hm3) ? yh[2 * i + 0] : U(0)) +
            ((h1 & hm3) ? yh[2 * i + 1] : U(0));
        accH[3] += ((h0 & hm4) ? yh[2 * i + 8] : U(0)) +
            ((h1 & hm4) ? yh[2 * i + 9] : U(0));
      }

      const U d = U(kq_q5_k_d(sb_addr));
      const U dmin = U(kq_q5_k_dmin(sb_addr));
      result[row] += d *
              (U(sc8[0]) *
                   ((acc1[0] + acc1[1] * (U(1) / U(256))) + U(16) * accH[0]) +
               U(sc8[1]) *
                   ((acc1[2] + acc1[3] * (U(1) / U(256))) * (U(1) / U(16)) +
                    U(16) * accH[1]) +
               U(sc8[4]) *
                   ((acc2[0] + acc2[1] * (U(1) / U(256))) + U(16) * accH[2]) +
               U(sc8[5]) *
                   ((acc2[2] + acc2[3] * (U(1) / U(256))) * (U(1) / U(16)) +
                    U(16) * accH[3])) -
          dmin *
              (sumy[0] * U(sc8[2]) + sumy[1] * U(sc8[3]) + sumy[2] * U(sc8[6]) +
               sumy[3] * U(sc8[7]));
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
struct KqQ5_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q5_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q5_K_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;

  static_assert(
      BCOLS == sub_block_size,
      "Q5_K loader requires BCOLS == 32 (one sub-block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_sub_block_idx;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short sub_block_idx;
  struct Cache {
    T vals[n_reads];
    uint8_t qh[n_reads];
  };
  metal::conditional_t<reduction_dim == 1, Cache, kq_empty> cached;

  KqQ5_KBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      if (sub_block_idx & 1) {
#pragma unroll
        for (short i = 0; i < n_reads; i++) {
          dst[i] = cached.vals[i];
        }
        return;
      }
    }

    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;

    const float d = float(*(const device half*)(src + KQ_Q5_K_D_OFFSET));
    const float dmin = float(*(const device half*)(src + KQ_Q5_K_DMIN_OFFSET));
    const device uint8_t* scales12 = src + KQ_Q5_K_SCALES_OFFSET;

    uint8_t sc6, mn6;
    kq_get_scale_min_k4(sb, scales12, sc6, mn6);
    const float eff_scale = d * float(sc6);
    const float eff_min = dmin * float(mn6);

    const short pair = sb / 2;
    const device uint8_t* qs = src + KQ_Q5_K_QS_OFFSET + pair * 32 + bj;
    const device uint8_t* qh = src + KQ_Q5_K_QH_OFFSET + bj;

    static_assert(
        n_reads == 8 || n_reads == 16,
        "Q5_K ALU vector load supports n_reads=8 (uint2) or 16 (uint4).");
    uint8_t qs_b[n_reads];
    if constexpr (n_reads == 8) {
      const uint2 qs_v = *reinterpret_cast<const device uint2*>(qs);
      *reinterpret_cast<thread uint2*>(&qs_b[0]) = qs_v;
    } else {
      const uint4 qs_v = *reinterpret_cast<const device uint4*>(qs);
      *reinterpret_cast<thread uint4*>(&qs_b[0]) = qs_v;
    }

    if constexpr (reduction_dim == 1) {
      uint8_t sc6_hi, mn6_hi;
      kq_get_scale_min_k4(sb + 1, scales12, sc6_hi, mn6_hi);
      const float eff_scale_hi = d * float(sc6_hi);
      const float eff_min_hi = dmin * float(mn6_hi);

      if (sub_block_idx == 0) {
        if constexpr (n_reads == 8) {
          const uint2 qh_v = *reinterpret_cast<const device uint2*>(qh);
          *reinterpret_cast<thread uint2*>(&cached.qh[0]) = qh_v;
        } else {
          const uint4 qh_v = *reinterpret_cast<const device uint4*>(qh);
          *reinterpret_cast<thread uint4*>(&cached.qh[0]) = qh_v;
        }
      }

#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t b = qs_b[i];
        const uint8_t h = cached.qh[i];
        const uint8_t q4_lo = b & 0x0F;
        const uint8_t q4_hi = b >> 4;
        const uint8_t hi_lo = (h >> sb) & 1u;
        const uint8_t hi_hi = (h >> (sb + 1)) & 1u;
        const uint8_t q5_lo = q4_lo | (hi_lo << 4);
        const uint8_t q5_hi = q4_hi | (hi_hi << 4);
        dst[i] = T(eff_scale * float(q5_lo) - eff_min);
        cached.vals[i] = T(eff_scale_hi * float(q5_hi) - eff_min_hi);
      }
    } else {
      uint8_t qh_b[n_reads];
      if constexpr (n_reads == 8) {
        const uint2 qh_v = *reinterpret_cast<const device uint2*>(qh);
        *reinterpret_cast<thread uint2*>(&qh_b[0]) = qh_v;
      } else {
        const uint4 qh_v = *reinterpret_cast<const device uint4*>(qh);
        *reinterpret_cast<thread uint4*>(&qh_b[0]) = qh_v;
      }
      const bool is_high = (sb & 1) != 0;
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t q4 = is_high ? (qs_b[i] >> 4) : (qs_b[i] & 0x0F);
        const uint8_t hi = (qh_b[i] >> sb) & 1u;
        const uint8_t q5 = q4 | (hi << 4);
        dst[i] = T(eff_scale * float(q5) - eff_min);
      }
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      sub_block_idx++;
      if (sub_block_idx == sub_blocks_per_block) {
        sub_block_idx = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q5_k_qmm_t(
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
      group_size == KQ_Q5_K_SUPERBLOCK, "Q5_K kernel requires group_size=256");
  static_assert(bits == 5, "Q5_K kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_KBlockLoader<
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
[[kernel]] void kq_q5_k_qmm_t_splitk(
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
      group_size == KQ_Q5_K_SUPERBLOCK, "Q5_K kernel requires group_size=256");
  static_assert(bits == 5, "Q5_K kernel requires bits=5");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_KBlockLoader<
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
[[kernel]] void kq_q5_k_qmm_n(
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
      group_size == KQ_Q5_K_SUPERBLOCK, "Q5_K kernel requires group_size=256");
  static_assert(bits == 5, "Q5_K kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ5_KBlockLoader<
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
[[kernel]] void kq_q5_k_qmv_fast(
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
  kq_q5_k_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_k_qmv(
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
  kq_q5_k_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_k_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q5_K_SUPERBLOCK, "Q5_K kernel requires group_size=256");
  static_assert(bits == 5, "Q5_K kernel requires bits=5");
  kq_q5_k_dequantize_impl<T>(w, out, num_weights, gid);
}

// Q6_K: 210 bytes/256 weights. REVERSED field order: [ql[128]][qh[64]][int8
// scales[16]][fp16 d]. q6 = (low4 | (high2<<4)) - 32; w[i] = d * sc * q6.

MLX_MTL_CONST int KQ_Q6_K_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_Q6_K_BLOCK_BYTES = 210;
MLX_MTL_CONST int KQ_Q6_K_QL_OFFSET = 0;
MLX_MTL_CONST int KQ_Q6_K_QH_OFFSET = 128;
MLX_MTL_CONST int KQ_Q6_K_SCALES_OFFSET = 192;
MLX_MTL_CONST int KQ_Q6_K_D_OFFSET = 208;

inline float kq_q6_k_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q6_K_D_OFFSET));
}
inline const device uint8_t* kq_q6_k_ql_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q6_K_QL_OFFSET;
}
inline const device uint8_t* kq_q6_k_qh_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q6_K_QH_OFFSET;
}
inline const device int8_t* kq_q6_k_scales_ptr(
    const device uint8_t* block_addr) {
  return (const device int8_t*)(block_addr + KQ_Q6_K_SCALES_OFFSET);
}

template <typename T>
METAL_FUNC void kq_q6_k_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_Q6_K_SUPERBLOCK;
  const int within_sb = gid - sb_id * KQ_Q6_K_SUPERBLOCK;

  const int half_idx = within_sb / 128;
  const int within_half = within_sb - half_idx * 128;
  const int quadrant = within_half / 32;
  const int l = within_half - quadrant * 32;

  const device uint8_t* sb_addr = w + sb_id * KQ_Q6_K_BLOCK_BYTES;
  const float d = kq_q6_k_d(sb_addr);
  const device uint8_t* ql = kq_q6_k_ql_ptr(sb_addr) + half_idx * 64;
  const device uint8_t* qh = kq_q6_k_qh_ptr(sb_addr) + half_idx * 32;
  const device int8_t* sc = kq_q6_k_scales_ptr(sb_addr) + half_idx * 8;

  const int ql_idx = (quadrant & 1) * 32 + l;
  const bool is_high_nibble = (quadrant >= 2);
  const uint8_t low4 = is_high_nibble ? (uint8_t)(ql[ql_idx] >> 4)
                                      : (uint8_t)(ql[ql_idx] & 0x0F);
  const uint8_t high2 = (uint8_t)((qh[l] >> (quadrant * 2)) & 0x03);
  const int8_t q6 = (int8_t)(low4 | (high2 << 4)) - (int8_t)32;
  const int is_off = l / 16;
  const int8_t scale_i8 = sc[is_off + 2 * quadrant];
  out[gid] = T(d * float(scale_i8) * float(q6));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q6_k_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q6_K_SUPERBLOCK, "Q6_K kernel requires group_size=256");
  static_assert(bits == 6, "Q6_K kernel requires bits=6");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int sb_stride = 2;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 2;
  const int ix = simd_lid % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;

  const int row_bytes = in_vec_size * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q6_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      yl[4 * l + 0] = U(x[x_base + l + 0]);
      yl[4 * l + 1] = U(x[x_base + l + 32]);
      yl[4 * l + 2] = U(x[x_base + l + 64]);
      yl[4 * l + 3] = U(x[x_base + l + 96]);
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;

      const device uint8_t* q1 = kq_q6_k_ql_ptr(sb_addr) + 64 * ip + l0;
      const device uint8_t* q2 = q1 + 32;
      const device uint8_t* qh = kq_q6_k_qh_ptr(sb_addr) + 32 * ip + l0;
      const device int8_t* sc = kq_q6_k_scales_ptr(sb_addr) + is;

      U sums[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int l = 0; l < 4; l++) {
        const uint8_t q1l = q1[l];
        const uint8_t q2l = q2[l];
        const uint8_t qhl = qh[l];
        const int8_t v0 =
            int8_t((q1l & 0x0F) | ((qhl & 0x03) << 4)) - int8_t(32);
        const int8_t v1 =
            int8_t((q2l & 0x0F) | ((qhl & 0x0C) << 2)) - int8_t(32);
        const int8_t v2 = int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
        const int8_t v3 = int8_t((q2l >> 4) | ((qhl & 0xC0) >> 2)) - int8_t(32);
        sums[0] += yl[4 * l + 0] * U(v0);
        sums[1] += yl[4 * l + 1] * U(v1);
        sums[2] += yl[4 * l + 2] * U(v2);
        sums[3] += yl[4 * l + 3] * U(v3);
      }

      const U d = U(kq_q6_k_d(sb_addr));
      result[row] += d *
          (sums[0] * U(sc[0]) + sums[1] * U(sc[2]) + sums[2] * U(sc[4]) +
           sums[3] * U(sc[6]));
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
METAL_FUNC void kq_q6_k_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q6_K_SUPERBLOCK, "Q6_K kernel requires group_size=256");
  static_assert(bits == 6, "Q6_K kernel requires bits=6");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int sb_stride = 2;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int tid_lane = simd_lid / 2;
  const int ix = simd_lid % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;

  const int nb = in_vec_size / KQ_Q6_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      yl[4 * l + 0] = U(x[x_base + l + 0]);
      yl[4 * l + 1] = U(x[x_base + l + 32]);
      yl[4 * l + 2] = U(x[x_base + l + 64]);
      yl[4 * l + 3] = U(x[x_base + l + 96]);
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;

      const device uint8_t* q1 = kq_q6_k_ql_ptr(sb_addr) + 64 * ip + l0;
      const device uint8_t* q2 = q1 + 32;
      const device uint8_t* qh = kq_q6_k_qh_ptr(sb_addr) + 32 * ip + l0;
      const device int8_t* sc = kq_q6_k_scales_ptr(sb_addr) + is;

      U sums[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int l = 0; l < 4; l++) {
        const uint8_t q1l = q1[l];
        const uint8_t q2l = q2[l];
        const uint8_t qhl = qh[l];
        const int8_t v0 =
            int8_t((q1l & 0x0F) | ((qhl & 0x03) << 4)) - int8_t(32);
        const int8_t v1 =
            int8_t((q2l & 0x0F) | ((qhl & 0x0C) << 2)) - int8_t(32);
        const int8_t v2 = int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
        const int8_t v3 = int8_t((q2l >> 4) | ((qhl & 0xC0) >> 2)) - int8_t(32);
        sums[0] += yl[4 * l + 0] * U(v0);
        sums[1] += yl[4 * l + 1] * U(v1);
        sums[2] += yl[4 * l + 2] * U(v2);
        sums[3] += yl[4 * l + 3] * U(v3);
      }

      const U d = U(kq_q6_k_d(sb_addr));
      result[row] += d *
          (sums[0] * U(sc[0]) + sums[1] * U(sc[2]) + sums[2] * U(sc[4]) +
           sums[3] * U(sc[6]));
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
struct KqQ6_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q6_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q6_K_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(
      BCOLS == k_tile_size,
      "Q6_K loader requires BCOLS == 32 (one K-tile per iteration).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt;
  struct Caches {
    T q1[n_reads];
    T q2[n_reads];
    T q3[n_reads];
  };
  metal::conditional_t<reduction_dim == 1, Caches, kq_empty> cached;

  KqQ6_KBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_kt(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      const short q = kt & 3;
      if (q == 1) {
#pragma unroll
        for (short i = 0; i < n_reads; i++)
          dst[i] = cached.q1[i];
        return;
      }
      if (q == 2) {
#pragma unroll
        for (short i = 0; i < n_reads; i++)
          dst[i] = cached.q2[i];
        return;
      }
      if (q == 3) {
#pragma unroll
        for (short i = 0; i < n_reads; i++)
          dst[i] = cached.q3[i];
        return;
      }
      const short half_idx = kt / 4;
      const short scale_off = (bj >= 16) ? 1 : 0;
      const float d = float(*(const device half*)(src + KQ_Q6_K_D_OFFSET));
      const device int8_t* scales =
          (const device int8_t*)(src + KQ_Q6_K_SCALES_OFFSET);
      const float es0 = d * float(scales[(kt + 0) * 2 + scale_off]);
      const float es1 = d * float(scales[(kt + 1) * 2 + scale_off]);
      const float es2 = d * float(scales[(kt + 2) * 2 + scale_off]);
      const float es3 = d * float(scales[(kt + 3) * 2 + scale_off]);

      const device uint8_t* ql_a =
          src + KQ_Q6_K_QL_OFFSET + half_idx * 64 + bj; // q=0 lo, q=2 hi
      const device uint8_t* ql_b =
          src + KQ_Q6_K_QL_OFFSET + half_idx * 64 + 32 + bj; // q=1 lo, q=3 hi
      const device uint8_t* qh = src + KQ_Q6_K_QH_OFFSET + half_idx * 32 + bj;

#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t a = ql_a[i];
        const uint8_t b = ql_b[i];
        const uint8_t h = qh[i];
        const int8_t q6_0 =
            (int8_t)((a & 0x0F) | ((h & 0x03) << 4)) - (int8_t)32;
        const int8_t q6_1 =
            (int8_t)((b & 0x0F) | (((h >> 2) & 0x03) << 4)) - (int8_t)32;
        const int8_t q6_2 =
            (int8_t)((a >> 4) | (((h >> 4) & 0x03) << 4)) - (int8_t)32;
        const int8_t q6_3 =
            (int8_t)((b >> 4) | (((h >> 6) & 0x03) << 4)) - (int8_t)32;
        dst[i] = T(es0 * float(q6_0));
        cached.q1[i] = T(es1 * float(q6_1));
        cached.q2[i] = T(es2 * float(q6_2));
        cached.q3[i] = T(es3 * float(q6_3));
      }
      return;
    }

    const short kt_use = fixed_kt;
    const short half_idx = kt_use / 4;
    const short quadrant = kt_use - half_idx * 4;
    const bool is_high_nibble = (quadrant >= 2);
    const short qh_shift = quadrant * 2;
    const short scale_idx = kt_use * 2 + (bj >= 16 ? 1 : 0);

    const float d = float(*(const device half*)(src + KQ_Q6_K_D_OFFSET));
    const int8_t scale_i8 =
        ((const device int8_t*)(src + KQ_Q6_K_SCALES_OFFSET))[scale_idx];
    const float eff_scale = d * float(scale_i8);

    const device uint8_t* ql_base =
        src + KQ_Q6_K_QL_OFFSET + half_idx * 64 + (quadrant & 1) * 32 + bj;
    const device uint8_t* qh_base =
        src + KQ_Q6_K_QH_OFFSET + half_idx * 32 + bj;

#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const uint8_t low4 =
          is_high_nibble ? (ql_base[i] >> 4) : (ql_base[i] & 0x0F);
      const uint8_t high2 = (uint8_t)((qh_base[i] >> qh_shift) & 0x03);
      const int8_t q6 = (int8_t)(low4 | (high2 << 4)) - (int8_t)32;
      dst[i] = T(eff_scale * float(q6));
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      kt++;
      if (kt == k_tiles_per_block) {
        kt = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q6_k_qmm_t(
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
      group_size == KQ_Q6_K_SUPERBLOCK, "Q6_K kernel requires group_size=256");
  static_assert(bits == 6, "Q6_K kernel requires bits=6");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ6_KBlockLoader<
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
[[kernel]] void kq_q6_k_qmm_t_splitk(
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
      group_size == KQ_Q6_K_SUPERBLOCK, "Q6_K kernel requires group_size=256");
  static_assert(bits == 6, "Q6_K kernel requires bits=6");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ6_KBlockLoader<
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
[[kernel]] void kq_q6_k_qmm_n(
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
      group_size == KQ_Q6_K_SUPERBLOCK, "Q6_K kernel requires group_size=256");
  static_assert(bits == 6, "Q6_K kernel requires bits=6");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ6_KBlockLoader<
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
[[kernel]] void kq_q6_k_qmv_fast(
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
  kq_q6_k_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q6_k_qmv(
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
  kq_q6_k_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q6_k_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q6_K_SUPERBLOCK, "Q6_K kernel requires group_size=256");
  static_assert(bits == 6, "Q6_K kernel requires bits=6");
  kq_q6_k_dequantize_impl<T>(w, out, num_weights, gid);
}

// Q3_K: 110 bytes/256 weights. [hmask[32]][qs[64]][scales[12]][fp16 d].
// q3 = q2 - h; hmask SET means h=0, CLEAR means h=4.
// w[i] = d * (scale - 32) * q3. Symmetric.

MLX_MTL_CONST int KQ_Q3_K_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_Q3_K_BLOCK_BYTES = 110;
MLX_MTL_CONST int KQ_Q3_K_HMASK_OFFSET = 0;
MLX_MTL_CONST int KQ_Q3_K_QS_OFFSET = 32;
MLX_MTL_CONST int KQ_Q3_K_SCALES_OFFSET = 96;
MLX_MTL_CONST int KQ_Q3_K_D_OFFSET = 108;

inline float kq_q3_k_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q3_K_D_OFFSET));
}
inline const device uint8_t* kq_q3_k_hmask_ptr(
    const device uint8_t* block_addr) {
  return block_addr + KQ_Q3_K_HMASK_OFFSET;
}
inline const device uint8_t* kq_q3_k_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q3_K_QS_OFFSET;
}
inline const device uint8_t* kq_q3_k_scales12_ptr(
    const device uint8_t* block_addr) {
  return block_addr + KQ_Q3_K_SCALES_OFFSET;
}

inline uint8_t kq_q3_k_unpack_scale(int j, const device uint8_t* q12) {
  const int quad = j / 4;
  const int byte = j & 3;
  const uint8_t low4 = (q12[(quad & 1) * 4 + byte] >> ((quad >> 1) * 4)) & 0x0F;
  const uint8_t high2 = (q12[8 + byte] >> (quad * 2)) & 0x03;
  return (uint8_t)(low4 | (high2 << 4));
}

template <typename T>
METAL_FUNC void kq_q3_k_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_Q3_K_SUPERBLOCK;
  const int within_sb = gid - sb_id * KQ_Q3_K_SUPERBLOCK;

  const int outer_half = within_sb / 128;
  const int within_outer = within_sb - outer_half * 128;
  const int shift_idx = within_outer / 32;
  const int within_shift = within_outer - shift_idx * 32;

  const device uint8_t* sb_addr = w + sb_id * KQ_Q3_K_BLOCK_BYTES;
  const float d = kq_q3_k_d(sb_addr);
  const int scale_idx = within_sb / 16;
  const uint8_t sc_unsigned =
      kq_q3_k_unpack_scale(scale_idx, kq_q3_k_scales12_ptr(sb_addr));
  const float eff_scale = d * float((int)sc_unsigned - 32);

  const uint8_t qs_byte =
      kq_q3_k_qs_ptr(sb_addr)[outer_half * 32 + within_shift];
  const uint8_t q2 = (qs_byte >> (shift_idx * 2)) & 0x03;
  const int hmask_bit = outer_half * 4 + shift_idx;
  const uint8_t hmask_byte = kq_q3_k_hmask_ptr(sb_addr)[within_shift];
  const bool hbit_set = ((hmask_byte >> hmask_bit) & 1) != 0;
  const int q3 = (int)q2 - (hbit_set ? 0 : 4);
  out[gid] = T(eff_scale * float(q3));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q3_k_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q3_K_SUPERBLOCK, "Q3_K kernel requires group_size=256");
  static_assert(bits == 3, "Q3_K kernel requires bits=3");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;

  const ushort4 mm_table[4] = {
      {0x0001, 0x0100, 0x0002, 0x0200},
      {0x0004, 0x0400, 0x0008, 0x0800},
      {0x0010, 0x1000, 0x0020, 0x2000},
      {0x0040, 0x4000, 0x0080, 0x8000},
  };
  const ushort4 qm_table[2] = {
      {0x0003, 0x0300, 0x000c, 0x0c00},
      {0x0030, 0x3000, 0x00c0, 0xc000},
  };

  typedef float U;
  thread U yl[32];
  thread U sumf1[results_per_simdgroup] = {0};
  thread U sumf2[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 4;
  const int ix = simd_lid % 4;
  const int ip = tid_lane / 4;
  const int il = 2 * ((tid_lane % 4) / 2);
  const int ir = tid_lane % 2;
  const int l0 = 8 * ir;
  const int tid_group = 2 * ip + il / 2;

  const ushort4 hm = mm_table[tid_group];
  const ushort4 qm = qm_table[il / 2];
  const int shift = 2 * il;
  const U v1 = (il == 0) ? U(4) : U(64);
  const U v2 = U(4) * v1;
  const uint16_t s_shift1 = uint16_t(4 * ip);
  const uint16_t s_shift2 = uint16_t(s_shift1 + il);

  const int row_bytes = in_vec_size * KQ_Q3_K_BLOCK_BYTES / KQ_Q3_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int q_offset_bytes = 32 * ip + l0;
  const int y_offset = 128 * ip + 32 * il + l0;
  const int nb = in_vec_size / KQ_Q3_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q3_K_SUPERBLOCK + y_offset;
#pragma unroll
    for (int l = 0; l < 8; l++) {
      yl[l + 0] = U(x[x_base + l + 0]);
      yl[l + 8] = U(x[x_base + l + 16]);
      yl[l + 16] = U(x[x_base + l + 32]);
      yl[l + 24] = U(x[x_base + l + 48]);
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q3_K_BLOCK_BYTES;

      const device uint16_t* q = reinterpret_cast<const device uint16_t*>(
          kq_q3_k_qs_ptr(sb_addr) + q_offset_bytes);
      const device uint16_t* h = reinterpret_cast<const device uint16_t*>(
          kq_q3_k_hmask_ptr(sb_addr) + l0);
      const device uint16_t* a = reinterpret_cast<const device uint16_t*>(
          kq_q3_k_scales12_ptr(sb_addr));

      uint32_t scales32, aux32;
      thread uint16_t* scales16 = reinterpret_cast<thread uint16_t*>(&scales32);
      thread const int8_t* scales =
          reinterpret_cast<thread const int8_t*>(&scales32);

      scales16[0] = a[4];
      scales16[1] = a[5];
      aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030u;
      scales16[0] = a[il + 0];
      scales16[1] = a[il + 1];
      scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0fu) | aux32;

      const U d_all = U(kq_q3_k_d(sb_addr));

      U s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
#pragma unroll
      for (int l = 0; l < 8; l += 2) {
        const uint16_t qs = q[l / 2];
        s1 += yl[l + 0] * U(qs & qm[0]);
        s2 += yl[l + 1] * U(qs & qm[1]);
        s3 += ((h[l / 2] & hm[0]) ? U(0) : yl[l + 0]) +
            ((h[l / 2] & hm[1]) ? U(0) : yl[l + 1]);
        s4 += yl[l + 16] * U(qs & qm[2]);
        s5 += yl[l + 17] * U(qs & qm[3]);
        s6 += ((h[l / 2] & hm[2]) ? U(0) : yl[l + 16]) +
            ((h[l / 2] & hm[3]) ? U(0) : yl[l + 17]);
      }
      U d1 = d_all * (s1 + s2 * (U(1) / U(256)) - s3 * v1);
      U d2 = d_all * (s4 + s5 * (U(1) / U(256)) - s6 * v2);
      sumf1[row] += d1 * U(int(scales[0]) - 32);
      sumf2[row] += d2 * U(int(scales[2]) - 32);

      s1 = s2 = s3 = s4 = s5 = s6 = U(0);
#pragma unroll
      for (int l = 0; l < 8; l += 2) {
        const uint16_t qs = q[l / 2 + 8];
        s1 += yl[l + 8] * U(qs & qm[0]);
        s2 += yl[l + 9] * U(qs & qm[1]);
        s3 += ((h[l / 2 + 8] & hm[0]) ? U(0) : yl[l + 8]) +
            ((h[l / 2 + 8] & hm[1]) ? U(0) : yl[l + 9]);
        s4 += yl[l + 24] * U(qs & qm[2]);
        s5 += yl[l + 25] * U(qs & qm[3]);
        s6 += ((h[l / 2 + 8] & hm[2]) ? U(0) : yl[l + 24]) +
            ((h[l / 2 + 8] & hm[3]) ? U(0) : yl[l + 25]);
      }
      d1 = d_all * (s1 + s2 * (U(1) / U(256)) - s3 * v1);
      d2 = d_all * (s4 + s5 * (U(1) / U(256)) - s6 * v2);
      sumf1[row] += d1 * U(int(scales[1]) - 32);
      sumf2[row] += d2 * U(int(scales[3]) - 32);
    }
  }

  const U shift_div = U(1) / U(1u << shift);
  for (int row = 0; row < results_per_simdgroup; row++) {
    const U combined = (sumf1[row] + U(0.25) * sumf2[row]) * shift_div;
    const U reduced = simd_sum(combined);
    if (simd_lid == 0) {
      y[out_row + row] = static_cast<T>(reduced);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q3_k_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q3_K_SUPERBLOCK, "Q3_K kernel requires group_size=256");
  static_assert(bits == 3, "Q3_K kernel requires bits=3");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;

  const ushort4 mm_table[4] = {
      {0x0001, 0x0100, 0x0002, 0x0200},
      {0x0004, 0x0400, 0x0008, 0x0800},
      {0x0010, 0x1000, 0x0020, 0x2000},
      {0x0040, 0x4000, 0x0080, 0x8000},
  };
  const ushort4 qm_table[2] = {
      {0x0003, 0x0300, 0x000c, 0x0c00},
      {0x0030, 0x3000, 0x00c0, 0xc000},
  };

  typedef float U;
  thread U yl[32];
  thread U sumf1[results_per_simdgroup] = {0};
  thread U sumf2[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q3_K_BLOCK_BYTES / KQ_Q3_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int tid_lane = simd_lid / 4;
  const int ix = simd_lid % 4;
  const int ip = tid_lane / 4;
  const int il = 2 * ((tid_lane % 4) / 2);
  const int ir = tid_lane % 2;
  const int l0 = 8 * ir;
  const int tid_group = 2 * ip + il / 2;

  const ushort4 hm = mm_table[tid_group];
  const ushort4 qm = qm_table[il / 2];
  const int shift = 2 * il;
  const U v1 = (il == 0) ? U(4) : U(64);
  const U v2 = U(4) * v1;
  const uint16_t s_shift1 = uint16_t(4 * ip);
  const uint16_t s_shift2 = uint16_t(s_shift1 + il);

  const int q_offset_bytes = 32 * ip + l0;
  const int y_offset = 128 * ip + 32 * il + l0;
  const int nb = in_vec_size / KQ_Q3_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q3_K_SUPERBLOCK + y_offset;
#pragma unroll
    for (int l = 0; l < 8; l++) {
      yl[l + 0] = U(x[x_base + l + 0]);
      yl[l + 8] = U(x[x_base + l + 16]);
      yl[l + 16] = U(x[x_base + l + 32]);
      yl[l + 24] = U(x[x_base + l + 48]);
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q3_K_BLOCK_BYTES;

      const device uint16_t* q = reinterpret_cast<const device uint16_t*>(
          kq_q3_k_qs_ptr(sb_addr) + q_offset_bytes);
      const device uint16_t* h = reinterpret_cast<const device uint16_t*>(
          kq_q3_k_hmask_ptr(sb_addr) + l0);
      const device uint16_t* a = reinterpret_cast<const device uint16_t*>(
          kq_q3_k_scales12_ptr(sb_addr));

      uint32_t scales32, aux32;
      thread uint16_t* scales16 = reinterpret_cast<thread uint16_t*>(&scales32);
      thread const int8_t* scales =
          reinterpret_cast<thread const int8_t*>(&scales32);

      scales16[0] = a[4];
      scales16[1] = a[5];
      aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030u;
      scales16[0] = a[il + 0];
      scales16[1] = a[il + 1];
      scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0fu) | aux32;

      const U d_all = U(kq_q3_k_d(sb_addr));

      U s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
#pragma unroll
      for (int l = 0; l < 8; l += 2) {
        const uint16_t qs = q[l / 2];
        s1 += yl[l + 0] * U(qs & qm[0]);
        s2 += yl[l + 1] * U(qs & qm[1]);
        s3 += ((h[l / 2] & hm[0]) ? U(0) : yl[l + 0]) +
            ((h[l / 2] & hm[1]) ? U(0) : yl[l + 1]);
        s4 += yl[l + 16] * U(qs & qm[2]);
        s5 += yl[l + 17] * U(qs & qm[3]);
        s6 += ((h[l / 2] & hm[2]) ? U(0) : yl[l + 16]) +
            ((h[l / 2] & hm[3]) ? U(0) : yl[l + 17]);
      }
      U d1 = d_all * (s1 + s2 * (U(1) / U(256)) - s3 * v1);
      U d2 = d_all * (s4 + s5 * (U(1) / U(256)) - s6 * v2);
      sumf1[row] += d1 * U(int(scales[0]) - 32);
      sumf2[row] += d2 * U(int(scales[2]) - 32);

      s1 = s2 = s3 = s4 = s5 = s6 = U(0);
#pragma unroll
      for (int l = 0; l < 8; l += 2) {
        const uint16_t qs = q[l / 2 + 8];
        s1 += yl[l + 8] * U(qs & qm[0]);
        s2 += yl[l + 9] * U(qs & qm[1]);
        s3 += ((h[l / 2 + 8] & hm[0]) ? U(0) : yl[l + 8]) +
            ((h[l / 2 + 8] & hm[1]) ? U(0) : yl[l + 9]);
        s4 += yl[l + 24] * U(qs & qm[2]);
        s5 += yl[l + 25] * U(qs & qm[3]);
        s6 += ((h[l / 2 + 8] & hm[2]) ? U(0) : yl[l + 24]) +
            ((h[l / 2 + 8] & hm[3]) ? U(0) : yl[l + 25]);
      }
      d1 = d_all * (s1 + s2 * (U(1) / U(256)) - s3 * v1);
      d2 = d_all * (s4 + s5 * (U(1) / U(256)) - s6 * v2);
      sumf1[row] += d1 * U(int(scales[1]) - 32);
      sumf2[row] += d2 * U(int(scales[3]) - 32);
    }
  }

  const U shift_div = U(1) / U(1u << shift);
  for (int row = 0; row < results_per_simdgroup; row++) {
    const U combined = (sumf1[row] + U(0.25) * sumf2[row]) * shift_div;
    const U reduced = simd_sum(combined);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(reduced);
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
struct KqQ3_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q3_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q3_K_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(
      BCOLS == k_tile_size,
      "Q3_K loader requires BCOLS == 32 (one K-tile per iteration).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt;
  struct Caches {
    T c1[n_reads];
    T c2[n_reads];
    T c3[n_reads];
    T c4[n_reads];
    T c5[n_reads];
    T c6[n_reads];
    T c7[n_reads];
  };
  metal::conditional_t<reduction_dim == 1, Caches, kq_empty> cached;

  KqQ3_KBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_kt(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      if (kt != 0) {
        if (kt == 1) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c1[i];
        } else if (kt == 2) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c2[i];
        } else if (kt == 3) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c3[i];
        } else if (kt == 4) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c4[i];
        } else if (kt == 5) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c5[i];
        } else if (kt == 6) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c6[i];
        } else {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c7[i];
        }
        return;
      }

      const float d = float(*(const device half*)(src + KQ_Q3_K_D_OFFSET));
      const short scale_off = (bj >= 16) ? 1 : 0;
      float es[8];
#pragma unroll
      for (short k = 0; k < 8; k++) {
        const uint8_t sc = kq_q3_k_unpack_scale(
            k * 2 + scale_off, src + KQ_Q3_K_SCALES_OFFSET);
        es[k] = d * float((int)sc - 32);
      }

      const device uint8_t* qs_a = src + KQ_Q3_K_QS_OFFSET + bj;
      const device uint8_t* qs_b = src + KQ_Q3_K_QS_OFFSET + 32 + bj;
      const device uint8_t* hm = src + KQ_Q3_K_HMASK_OFFSET + bj;

#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t qa = qs_a[i];
        const uint8_t qb = qs_b[i];
        const uint8_t h = hm[i];
        const uint8_t q2_0 = qa & 0x03;
        const uint8_t q2_1 = (qa >> 2) & 0x03;
        const uint8_t q2_2 = (qa >> 4) & 0x03;
        const uint8_t q2_3 = (qa >> 6) & 0x03;
        const uint8_t q2_4 = qb & 0x03;
        const uint8_t q2_5 = (qb >> 2) & 0x03;
        const uint8_t q2_6 = (qb >> 4) & 0x03;
        const uint8_t q2_7 = (qb >> 6) & 0x03;
        const int q3_0 = (int)q2_0 - (((h >> 0) & 1) ? 0 : 4);
        const int q3_1 = (int)q2_1 - (((h >> 1) & 1) ? 0 : 4);
        const int q3_2 = (int)q2_2 - (((h >> 2) & 1) ? 0 : 4);
        const int q3_3 = (int)q2_3 - (((h >> 3) & 1) ? 0 : 4);
        const int q3_4 = (int)q2_4 - (((h >> 4) & 1) ? 0 : 4);
        const int q3_5 = (int)q2_5 - (((h >> 5) & 1) ? 0 : 4);
        const int q3_6 = (int)q2_6 - (((h >> 6) & 1) ? 0 : 4);
        const int q3_7 = (int)q2_7 - (((h >> 7) & 1) ? 0 : 4);
        dst[i] = T(es[0] * float(q3_0));
        cached.c1[i] = T(es[1] * float(q3_1));
        cached.c2[i] = T(es[2] * float(q3_2));
        cached.c3[i] = T(es[3] * float(q3_3));
        cached.c4[i] = T(es[4] * float(q3_4));
        cached.c5[i] = T(es[5] * float(q3_5));
        cached.c6[i] = T(es[6] * float(q3_6));
        cached.c7[i] = T(es[7] * float(q3_7));
      }
      return;
    }

    const short kt_use = fixed_kt;
    const short outer_half = kt_use / 4;
    const short qs_shift = (kt_use & 3) * 2;
    const short hmask_bit = kt_use;
    const short scale_idx = kt_use * 2 + (bj >= 16 ? 1 : 0);

    const float d = float(*(const device half*)(src + KQ_Q3_K_D_OFFSET));
    const uint8_t sc_unsigned =
        kq_q3_k_unpack_scale(scale_idx, src + KQ_Q3_K_SCALES_OFFSET);
    const float eff_scale = d * float((int)sc_unsigned - 32);

    const device uint8_t* qs = src + KQ_Q3_K_QS_OFFSET + outer_half * 32 + bj;
    const device uint8_t* hm = src + KQ_Q3_K_HMASK_OFFSET + bj;

#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const uint8_t q2 = (qs[i] >> qs_shift) & 0x03;
      const bool hbit_set = ((hm[i] >> hmask_bit) & 1) != 0;
      const int q3 = (int)q2 - (hbit_set ? 0 : 4);
      dst[i] = T(eff_scale * float(q3));
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      kt++;
      if (kt == k_tiles_per_block) {
        kt = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q3_k_qmm_t(
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
      group_size == KQ_Q3_K_SUPERBLOCK, "Q3_K kernel requires group_size=256");
  static_assert(bits == 3, "Q3_K kernel requires bits=3");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ3_KBlockLoader<
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
[[kernel]] void kq_q3_k_qmm_t_splitk(
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
      group_size == KQ_Q3_K_SUPERBLOCK, "Q3_K kernel requires group_size=256");
  static_assert(bits == 3, "Q3_K kernel requires bits=3");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ3_KBlockLoader<
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
[[kernel]] void kq_q3_k_qmm_n(
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
      group_size == KQ_Q3_K_SUPERBLOCK, "Q3_K kernel requires group_size=256");
  static_assert(bits == 3, "Q3_K kernel requires bits=3");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ3_KBlockLoader<
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
[[kernel]] void kq_q3_k_qmv_fast(
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
  kq_q3_k_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q3_k_qmv(
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
  kq_q3_k_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q3_k_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q3_K_SUPERBLOCK, "Q3_K kernel requires group_size=256");
  static_assert(bits == 3, "Q3_K kernel requires bits=3");
  kq_q3_k_dequantize_impl<T>(w, out, num_weights, gid);
}

// Q2_K: 84 bytes/256 weights. [scales[16]][qs[64]][fp16 d][fp16 dmin].
// w[i] = d * (sc & 0xF) * q2 - dmin * (sc >> 4). Asymmetric.

MLX_MTL_CONST int KQ_Q2_K_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_Q2_K_BLOCK_BYTES = 84;
MLX_MTL_CONST int KQ_Q2_K_SCALES_OFFSET = 0;
MLX_MTL_CONST int KQ_Q2_K_QS_OFFSET = 16;
MLX_MTL_CONST int KQ_Q2_K_D_OFFSET = 80;
MLX_MTL_CONST int KQ_Q2_K_DMIN_OFFSET = 82;

inline float kq_q2_k_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q2_K_D_OFFSET));
}
inline float kq_q2_k_dmin(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q2_K_DMIN_OFFSET));
}
inline const device uint8_t* kq_q2_k_scales_ptr(
    const device uint8_t* block_addr) {
  return block_addr + KQ_Q2_K_SCALES_OFFSET;
}
inline const device uint8_t* kq_q2_k_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q2_K_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q2_k_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_Q2_K_SUPERBLOCK;
  const int within_sb = gid - sb_id * KQ_Q2_K_SUPERBLOCK;

  const int outer_half = within_sb / 128;
  const int within_outer = within_sb - outer_half * 128;
  const int shift_idx = within_outer / 32;
  const int within_shift = within_outer - shift_idx * 32;

  const device uint8_t* sb_addr = w + sb_id * KQ_Q2_K_BLOCK_BYTES;
  const float d = kq_q2_k_d(sb_addr);
  const float dmin = kq_q2_k_dmin(sb_addr);
  const int scale_idx = within_sb / 16;
  const uint8_t sc_byte = kq_q2_k_scales_ptr(sb_addr)[scale_idx];
  const float eff_scale = d * float(sc_byte & 0x0F);
  const float eff_min = dmin * float(sc_byte >> 4);

  const uint8_t qs_byte =
      kq_q2_k_qs_ptr(sb_addr)[outer_half * 32 + within_shift];
  const uint8_t q2 = (qs_byte >> (shift_idx * 2)) & 0x03;
  out[gid] = T(eff_scale * float(q2) - eff_min);
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q2_k_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q2_K_SUPERBLOCK, "Q2_K kernel requires group_size=256");
  static_assert(bits == 2, "Q2_K kernel requires bits=2");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;

  typedef float U;
  thread U yl[32];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 8;
  const int it = simd_lid % 8;
  const int iq = it / 4;
  const int ir = it % 4;
  const int is = (8 * ir) / 16;

  const int row_bytes = in_vec_size * KQ_Q2_K_BLOCK_BYTES / KQ_Q2_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q2_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q2_K_SUPERBLOCK + 128 * iq + 8 * ir;
    U sumy[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      yl[i + 0] = U(x[x_base + i + 0]);
      sumy[0] += yl[i + 0];
      yl[i + 8] = U(x[x_base + i + 32]);
      sumy[1] += yl[i + 8];
      yl[i + 16] = U(x[x_base + i + 64]);
      sumy[2] += yl[i + 16];
      yl[i + 24] = U(x[x_base + i + 96]);
      sumy[3] += yl[i + 24];
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q2_K_BLOCK_BYTES;

      const device uint8_t* sc = kq_q2_k_scales_ptr(sb_addr) + 8 * iq + is;
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q2_k_qs_ptr(sb_addr)) +
          16 * iq + 4 * ir;

      U acc1[4] = {U(0), U(0), U(0), U(0)};
      U acc2[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qs_i = qs[i / 2];
        acc1[0] += yl[i + 0] * U(qs_i & 0x0003);
        acc2[0] += yl[i + 1] * U(qs_i & 0x0300);
        acc1[1] += yl[i + 8] * U(qs_i & 0x000c);
        acc2[1] += yl[i + 9] * U(qs_i & 0x0c00);
        acc1[2] += yl[i + 16] * U(qs_i & 0x0030);
        acc2[2] += yl[i + 17] * U(qs_i & 0x3000);
        acc1[3] += yl[i + 24] * U(qs_i & 0x00c0);
        acc2[3] += yl[i + 25] * U(qs_i & 0xc000);
      }

      const U d = U(kq_q2_k_d(sb_addr));
      const U dmin = U(kq_q2_k_dmin(sb_addr));
      result[row] += d *
              ((acc1[0] + acc2[0] * (U(1) / U(256))) * U(sc[0] & 0x0F) +
               (acc1[1] + acc2[1] * (U(1) / U(256))) * U(sc[2] & 0x0F) *
                   (U(1) / U(4)) +
               (acc1[2] + acc2[2] * (U(1) / U(256))) * U(sc[4] & 0x0F) *
                   (U(1) / U(16)) +
               (acc1[3] + acc2[3] * (U(1) / U(256))) * U(sc[6] & 0x0F) *
                   (U(1) / U(64))) -
          dmin * (U(1) / U(16)) *
              (sumy[0] * U(sc[0] & 0xF0) + sumy[1] * U(sc[2] & 0xF0) +
               sumy[2] * U(sc[4] & 0xF0) + sumy[3] * U(sc[6] & 0xF0));
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
METAL_FUNC void kq_q2_k_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q2_K_SUPERBLOCK, "Q2_K kernel requires group_size=256");
  static_assert(bits == 2, "Q2_K kernel requires bits=2");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;

  typedef float U;
  thread U yl[32];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q2_K_BLOCK_BYTES / KQ_Q2_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int ix = simd_lid / 8;
  const int it = simd_lid % 8;
  const int iq = it / 4;
  const int ir = it % 4;
  const int is = (8 * ir) / 16;

  const int nb = in_vec_size / KQ_Q2_K_SUPERBLOCK;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += sb_stride) {
    const int x_base = ib * KQ_Q2_K_SUPERBLOCK + 128 * iq + 8 * ir;
    U sumy[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      yl[i + 0] = U(x[x_base + i + 0]);
      sumy[0] += yl[i + 0];
      yl[i + 8] = U(x[x_base + i + 32]);
      sumy[1] += yl[i + 8];
      yl[i + 16] = U(x[x_base + i + 64]);
      sumy[2] += yl[i + 16];
      yl[i + 24] = U(x[x_base + i + 96]);
      sumy[3] += yl[i + 24];
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* sb_addr =
          w + row_idx * row_bytes + ib * KQ_Q2_K_BLOCK_BYTES;

      const device uint8_t* sc = kq_q2_k_scales_ptr(sb_addr) + 8 * iq + is;
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q2_k_qs_ptr(sb_addr)) +
          16 * iq + 4 * ir;

      U acc1[4] = {U(0), U(0), U(0), U(0)};
      U acc2[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qs_i = qs[i / 2];
        acc1[0] += yl[i + 0] * U(qs_i & 0x0003);
        acc2[0] += yl[i + 1] * U(qs_i & 0x0300);
        acc1[1] += yl[i + 8] * U(qs_i & 0x000c);
        acc2[1] += yl[i + 9] * U(qs_i & 0x0c00);
        acc1[2] += yl[i + 16] * U(qs_i & 0x0030);
        acc2[2] += yl[i + 17] * U(qs_i & 0x3000);
        acc1[3] += yl[i + 24] * U(qs_i & 0x00c0);
        acc2[3] += yl[i + 25] * U(qs_i & 0xc000);
      }

      const U d = U(kq_q2_k_d(sb_addr));
      const U dmin = U(kq_q2_k_dmin(sb_addr));
      result[row] += d *
              ((acc1[0] + acc2[0] * (U(1) / U(256))) * U(sc[0] & 0x0F) +
               (acc1[1] + acc2[1] * (U(1) / U(256))) * U(sc[2] & 0x0F) *
                   (U(1) / U(4)) +
               (acc1[2] + acc2[2] * (U(1) / U(256))) * U(sc[4] & 0x0F) *
                   (U(1) / U(16)) +
               (acc1[3] + acc2[3] * (U(1) / U(256))) * U(sc[6] & 0x0F) *
                   (U(1) / U(64))) -
          dmin * (U(1) / U(16)) *
              (sumy[0] * U(sc[0] & 0xF0) + sumy[1] * U(sc[2] & 0xF0) +
               sumy[2] * U(sc[4] & 0xF0) + sumy[3] * U(sc[6] & 0xF0));
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
struct KqQ2_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q2_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q2_K_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(
      BCOLS == k_tile_size,
      "Q2_K loader requires BCOLS == 32 (one K-tile per iteration).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt;
  struct Caches {
    T c1[n_reads];
    T c2[n_reads];
    T c3[n_reads];
    T c4[n_reads];
    T c5[n_reads];
    T c6[n_reads];
    T c7[n_reads];
  };
  metal::conditional_t<reduction_dim == 1, Caches, kq_empty> cached;

  KqQ2_KBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_kt(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      if (kt != 0) {
        if (kt == 1) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c1[i];
        } else if (kt == 2) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c2[i];
        } else if (kt == 3) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c3[i];
        } else if (kt == 4) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c4[i];
        } else if (kt == 5) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c5[i];
        } else if (kt == 6) {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c6[i];
        } else {
#pragma unroll
          for (short i = 0; i < n_reads; i++)
            dst[i] = cached.c7[i];
        }
        return;
      }

      const float d = float(*(const device half*)(src + KQ_Q2_K_D_OFFSET));
      const float dmin =
          float(*(const device half*)(src + KQ_Q2_K_DMIN_OFFSET));
      const short scale_off = (bj >= 16) ? 1 : 0;
      float es[8];
      float em[8];
#pragma unroll
      for (short k = 0; k < 8; k++) {
        const uint8_t sc_byte = src[KQ_Q2_K_SCALES_OFFSET + k * 2 + scale_off];
        es[k] = d * float(sc_byte & 0x0F);
        em[k] = dmin * float(sc_byte >> 4);
      }

      static_assert(
          n_reads == 8 || n_reads == 16,
          "Q2_K ALU vector load supports n_reads=8 or 16 (uint).");
      const device uint8_t* qs_a = src + KQ_Q2_K_QS_OFFSET + bj;
      const device uint8_t* qs_b = src + KQ_Q2_K_QS_OFFSET + 32 + bj;
      uint8_t qa_b[n_reads];
      uint8_t qb_b[n_reads];
#pragma unroll
      for (short v = 0; v < n_reads / 4; v++) {
        const uint qs_a_v = *reinterpret_cast<const device uint*>(qs_a + v * 4);
        const uint qs_b_v = *reinterpret_cast<const device uint*>(qs_b + v * 4);
        *reinterpret_cast<thread uint*>(&qa_b[v * 4]) = qs_a_v;
        *reinterpret_cast<thread uint*>(&qb_b[v * 4]) = qs_b_v;
      }

#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t qa = qa_b[i];
        const uint8_t qb = qb_b[i];
        const uint8_t q2_0 = qa & 0x03;
        const uint8_t q2_1 = (qa >> 2) & 0x03;
        const uint8_t q2_2 = (qa >> 4) & 0x03;
        const uint8_t q2_3 = (qa >> 6) & 0x03;
        const uint8_t q2_4 = qb & 0x03;
        const uint8_t q2_5 = (qb >> 2) & 0x03;
        const uint8_t q2_6 = (qb >> 4) & 0x03;
        const uint8_t q2_7 = (qb >> 6) & 0x03;
        dst[i] = T(es[0] * float(q2_0) - em[0]);
        cached.c1[i] = T(es[1] * float(q2_1) - em[1]);
        cached.c2[i] = T(es[2] * float(q2_2) - em[2]);
        cached.c3[i] = T(es[3] * float(q2_3) - em[3]);
        cached.c4[i] = T(es[4] * float(q2_4) - em[4]);
        cached.c5[i] = T(es[5] * float(q2_5) - em[5]);
        cached.c6[i] = T(es[6] * float(q2_6) - em[6]);
        cached.c7[i] = T(es[7] * float(q2_7) - em[7]);
      }
      return;
    }

    const short kt_use = fixed_kt;
    const short outer_half = kt_use / 4;
    const short qs_shift = (kt_use & 3) * 2;
    const short scale_idx = kt_use * 2 + (bj >= 16 ? 1 : 0);

    const float d = float(*(const device half*)(src + KQ_Q2_K_D_OFFSET));
    const float dmin = float(*(const device half*)(src + KQ_Q2_K_DMIN_OFFSET));
    const uint8_t sc_byte = src[KQ_Q2_K_SCALES_OFFSET + scale_idx];
    const float eff_scale = d * float(sc_byte & 0x0F);
    const float eff_min = dmin * float(sc_byte >> 4);

    const device uint8_t* qs = src + KQ_Q2_K_QS_OFFSET + outer_half * 32 + bj;

#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const uint8_t q2 = (qs[i] >> qs_shift) & 0x03;
      dst[i] = T(eff_scale * float(q2) - eff_min);
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      kt++;
      if (kt == k_tiles_per_block) {
        kt = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q2_k_qmm_t(
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
      group_size == KQ_Q2_K_SUPERBLOCK, "Q2_K kernel requires group_size=256");
  static_assert(bits == 2, "Q2_K kernel requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ2_KBlockLoader<
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
[[kernel]] void kq_q2_k_qmm_t_splitk(
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
      group_size == KQ_Q2_K_SUPERBLOCK, "Q2_K kernel requires group_size=256");
  static_assert(bits == 2, "Q2_K kernel requires bits=2");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ2_KBlockLoader<
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
[[kernel]] void kq_q2_k_qmm_n(
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
      group_size == KQ_Q2_K_SUPERBLOCK, "Q2_K kernel requires group_size=256");
  static_assert(bits == 2, "Q2_K kernel requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ2_KBlockLoader<
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
[[kernel]] void kq_q2_k_qmv_fast(
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
  kq_q2_k_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q2_k_qmv(
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
  kq_q2_k_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q2_k_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q2_K_SUPERBLOCK, "Q2_K kernel requires group_size=256");
  static_assert(bits == 2, "Q2_K kernel requires bits=2");
  kq_q2_k_dequantize_impl<T>(w, out, num_weights, gid);
}

#define KQUANT_DEFINE_GATHER_KERNELS(CODEC, LOADER)                   \
  template <typename T, int group_size, int bits>                     \
  [[kernel]] void kq_##CODEC##_gather_qmv_fast(                       \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& in_vec_size,                                \
      const constant int& out_vec_size,                               \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    int M = x_shape[x_batch_ndims];                                   \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        out_vec_size * M,                                             \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    kq_##CODEC##_qmv_fast_impl<T, group_size, bits>(                  \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid); \
  }                                                                   \
                                                                      \
  template <typename T, int group_size, int bits>                     \
  [[kernel]] void kq_##CODEC##_gather_qmv(                            \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& in_vec_size,                                \
      const constant int& out_vec_size,                               \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    int M = x_shape[x_batch_ndims];                                   \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        out_vec_size * M,                                             \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    kq_##CODEC##_qmv_impl<T, group_size, bits>(                       \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid); \
  }                                                                   \
                                                                      \
  template <typename T, int group_size, int bits, bool aligned_N>     \
  [[kernel]] void kq_##CODEC##_gather_qmm_t(                          \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& K,                                          \
      const constant int& N,                                          \
      const constant int& M,                                          \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint lid [[thread_index_in_threadgroup]],                       \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        M * N,                                                        \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    constexpr int BM = 32, BK = 32, BN = 32;                          \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                  \
    threadgroup T Xs[BM * BK_padded];                                 \
    threadgroup T Ws[BN * BK_padded];                                 \
    using LoaderW = LOADER<                                           \
        T,                                                            \
        BN,                                                           \
        BK,                                                           \
        BK_padded,                                                    \
        /*reduction_dim=*/1,                                          \
        /*tgp_size=*/2 * 2 * SIMD_SIZE>;                              \
    kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(                 \
        w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);   \
  }                                                                   \
                                                                      \
  template <typename T, int group_size, int bits>                     \
  [[kernel]] void kq_##CODEC##_gather_qmm_n(                          \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& K,                                          \
      const constant int& N,                                          \
      const constant int& M,                                          \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint lid [[thread_index_in_threadgroup]],                       \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        M * N,                                                        \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    constexpr int BM = 32, BK = 32, BN = 32;                          \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                  \
    constexpr int BN_padded = (BN + 16 / sizeof(T));                  \
    threadgroup T Xs[BM * BK_padded];                                 \
    threadgroup T Ws[BK * BN_padded];                                 \
    using LoaderW = LOADER<                                           \
        T,                                                            \
        BK,                                                           \
        BN,                                                           \
        BN_padded,                                                    \
        /*reduction_dim=*/0,                                          \
        /*tgp_size=*/2 * 2 * SIMD_SIZE>;                              \
    kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(                            \
        w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);      \
  }

KQUANT_DEFINE_GATHER_KERNELS(q8_0, KqQ8_0BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q4_0, KqQ4_0BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q4_1, KqQ4_1BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q5_0, KqQ5_0BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q5_1, KqQ5_1BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q4_k, KqQ4_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q5_k, KqQ5_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q6_k, KqQ6_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q3_k, KqQ3_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q2_k, KqQ2_KBlockLoader)

#undef KQUANT_DEFINE_GATHER_KERNELS
