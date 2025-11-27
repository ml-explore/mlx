// Copyright © 2025 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/fp4.h"
#include "mlx/backend/metal/kernels/fp8.h"

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int wsize = 8>
inline constexpr short get_pack_factor() {
  return wsize / 4;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T>
static inline T dequantize_scale(uint8_t s) {
  return T(*(thread fp8_e8m0*)(&s));
}

template <int bits>
struct Quantize {
  uint8_t operator()(float x) {
    if constexpr (bits == 8) {
      return fp8_e4m3(x).bits;
    } else {
      return fp4_e2m1(x).bits;
    }
  }
};

template <int bits>
struct Dequantize {
  float operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return float(*(thread fp8_e4m3*)(&x));
    } else {
      return float(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename U, int N>
inline void dequantize(
    const device uint8_t* w,
    U scale,
    threadgroup U* w_local,
    const threadgroup U* lut) {
  for (int i = 0; i < (N / 2); i++) {
    w_local[2 * i] = scale * lut[w[i] & 0xf];
    w_local[2 * i + 1] = scale * lut[(w[i] >> 4) & 0xf];
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS % group_size == 0,
      "The group size should be divisible by the columns");

  MLX_MTL_CONST short pack_factor = get_pack_factor<8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  static_assert(
      (BCOLS_PACKED / n_reads) == n_groups,
      "Other configurations are not yet supported");

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint8_t* scales;
  threadgroup T* lut;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      threadgroup T* lut_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id),
        lut(lut_) {
    if (simd_group_id == 0 && simd_lane_id < 16) {
      lut[simd_lane_id] = static_cast<T>(FP4_LUT[simd_lane_id]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = dequantize_scale<T>(*scales);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor>(
          src + i * bytes_per_pack, scale, dst + i * pack_factor, lut);
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

    T scale = dequantize_scale<T>(*scales);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          dst + i * pack_factor,
          lut);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      // if (group_steps > 1) {
      //   group_step_cnt++;
      //   if (group_step_cnt == group_steps) {
      //     group_step_cnt = 0;
      //     scales++;
      //   }
      // } else {
      scales += n_groups;
      // }
    } else {
      scales += n_groups * group_stride;
    }
  }
};

using namespace mlx::steel;

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
METAL_FUNC void fp_qmm_t_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    threadgroup Wtype* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    threadgroup Wtype* lut) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<8>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  // Instantiate Loader
  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, K, Ws, lut, simd_gid, simd_lid);

  constexpr short UM = 16;
  constexpr short UN = 32;
  constexpr short UK = 16;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / UM;
  constexpr short TN = SN / UN;
  constexpr short TK = SK / UK;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(SM, short(M - (y_row + tm)));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(SN, short(N - (y_col + tn)));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  using ASubTile = NAXSubTile<T, UM, UK>;
  using BSubTile = NAXSubTile<Wtype, UN, UK>;
  using DSubTile = NAXSubTile<AccumType, UM, UN>;

  NAXTile<AccumType, TM, TN, DSubTile> Dtile;

  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK, ASubTile> Atile;
          NAXTile<Wtype, TN, TK, BSubTile> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<Wtype, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
METAL_FUNC void fp_qmm_n_impl(
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
    uint simd_lid [[thread_index_in_simdgroup]],
    threadgroup T* lut) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<8>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size>;

  auto wl = (const device uint8_t*)w;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, N, Ws, lut, simd_gid, simd_lid);
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

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
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
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  y += tid.z * output_stride;
}

template <typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
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
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
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
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  y += tid.z * output_stride;
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_t_nax(
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

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  threadgroup Wtype Ws[BN * BK_padded];
  threadgroup Wtype lut[16];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }
  fp_qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid, lut);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_n_nax(
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
  threadgroup T lut[16];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }

  fp_qmm_n_impl<T, group_size, bits, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid, lut);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_t_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
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
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  threadgroup Wtype Ws[BN * BK_padded];
  threadgroup Wtype lut[16];

  adjust_matrix_offsets(
      x,
      w,
      scales,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid, lut);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_n_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
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
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  threadgroup T lut[16];

  adjust_matrix_offsets(
      x,
      w,
      scales,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_n_impl<T, group_size, bits, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid, lut);
}

template <
    typename T,
    int group_size,
    const int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_rhs_nax(
    const device T* x,
    const device uint32_t* w,
    const device uint8_t* scales,
    const device uint32_t* indices,
    device T* y,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<8>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));
  constexpr int BN_padded = (BN + 16 / sizeof(Wtype));

  threadgroup Wtype lut[16];

  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size>;

  threadgroup Wtype Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short UM = 16;
  constexpr short UN = 32;
  constexpr short UK = 16;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / UM;
  constexpr short TN = SN / UN;
  constexpr short TK = SK / UK;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm =
      align_M ? SM : min(SM, short(max(0, (M - (y_row + tm)))));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  using ASubTile = NAXSubTile<T, UM, UK>;
  using BSubTile = NAXSubTile<Wtype, transpose ? UN : UK, transpose ? UK : UN>;
  using DSubTile = NAXSubTile<AccumType, UM, UN>;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    NAXTile<AccumType, TM, TN, DSubTile> Dtile;

    Dtile.clear();

    const device T* xn = x + tm * K;

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        transpose ? K : N,
        Ws,
        lut,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK, ASubTile> Atile;
            NAXTile<Wtype, BR, BC, BSubTile> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(xn + kk1, K);
            } else {
              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            }

            if constexpr (transpose) {
              Btile.template load<Wtype, BK_padded, 1>(
                  Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<Wtype, BN_padded, 1>(
                  Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }

          xn += BK;
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK, ASubTile> Atile;
            NAXTile<Wtype, BR, BC, BSubTile> Btile;

            volatile int compiler_barrier;

            const short psk = min(int(SK), max(0, (BK - kk1)));
            Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

            if constexpr (transpose) {
              Btile.template load<Wtype, BK_padded, 1>(
                  Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<Wtype, BN_padded, 1>(
                  Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
        const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));

        // Store results to device memory
        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
}
