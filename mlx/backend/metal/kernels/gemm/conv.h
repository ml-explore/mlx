// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/conv_params.h"

#define MLX_MTL_CONST static constant constexpr const

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int vec_size,
    int tgp_size,
    int tgp_padding = 0>
struct Conv2DInputBlockLoader {
  // Destination dimensions
  MLX_MTL_CONST int dst_fd = BM;
  MLX_MTL_CONST int dst_ld = BK + tgp_padding;
  MLX_MTL_CONST int n_vecs = BK / vec_size;

  // Stride along block row within the block
  MLX_MTL_CONST int bstride = tgp_size / n_vecs;
  MLX_MTL_CONST int n_rows = dst_fd / bstride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>& params;

  int weight_h;
  int weight_w;

  int offsets_n[n_rows];
  int offsets_oh[n_rows];
  int offsets_ow[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoader(
      const device T* src_,
      threadgroup T* dst_,
      const constant MLXConvParams<2>& params_,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / n_vecs),
        bj(vec_size * (thread_idx % n_vecs)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bj),
        params(params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params.oS[0] * params.oS[1];

    for (int i = 0; i < n_rows; ++i) {
      int offset_nhw = tid.y * BM + bi + i * bstride;
      offsets_n[i] = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      offsets_oh[i] = hw / params.oS[1];
      offsets_ow[i] = hw % params.oS[1];
    }

    (void)lid;
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
#pragma clang loop unroll(full)
    for (short i = 0, is = 0; i < n_rows; ++i, is += bstride) {
      int n = offsets_n[i];
      int oh = offsets_oh[i];
      int ow = offsets_ow[i];

      int ih = oh * params.str[0] - params.pad[0] + weight_h * params.dil[0];
      int iw = ow * params.str[1] - params.pad[1] + weight_w * params.dil[1];

      // Read from input if in bounds
      if (ih >= 0 && ih < params.iS[0] && iw >= 0 && iw < params.iS[1]) {
        const device T* curr_src = src + n * params.in_strides[0] +
            ih * params.in_strides[1] + iw * params.in_strides[2];

#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = curr_src[j];
        }
      }

      // Zero pad otherwise
      else {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    if (++weight_w < params.wS[1]) {
      return;
    }

    weight_w = 0;

    if (++weight_h < params.wS[0]) {
      return;
    }

    weight_h = 0;

    src += BK;
  }
};

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int vec_size,
    int tgp_size,
    int tgp_padding = 0>
struct Conv2DWeightBlockLoader {
  // Destination dimensions
  MLX_MTL_CONST int dst_fd = BN;
  MLX_MTL_CONST int dst_ld = BK + tgp_padding;
  MLX_MTL_CONST int n_vecs = BK / vec_size;

  // Stride along block row within the block
  MLX_MTL_CONST int bstride = tgp_size / n_vecs;
  MLX_MTL_CONST int n_rows = dst_fd / bstride;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>& params;

  int weight_h;
  int weight_w;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoader(
      const device T* src_,
      threadgroup T* dst_,
      const constant MLXConvParams<2>& params_,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(params_.wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / n_vecs),
        bj(vec_size * (thread_idx % n_vecs)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        weight_h(0),
        weight_w(0) {
    (void)lid;
    (void)tid;
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    const device T* curr_src =
        src + weight_h * params.wt_strides[1] + weight_w * params.wt_strides[2];
#pragma clang loop unroll(full)
    for (short i = 0; i < dst_fd; i += bstride) {
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = curr_src[i * src_ld + j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    if (++weight_w < params.wS[1]) {
      return;
    }

    weight_w = 0;

    if (++weight_h < params.wS[0]) {
      return;
    }

    weight_h = 0;

    src += BK;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Transforms
///////////////////////////////////////////////////////////////////////////////

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    int tgp_padding_a = 0,
    int tgp_padding_b = 0,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<T, AccumType>>
struct Conv2DBlockMMA {
  // Warp tile size along M
  MLX_MTL_CONST int TM = BM / (WM * 8);
  // Warp tile size along N
  MLX_MTL_CONST int TN = BN / (WN * 8);

  // Warp tile simdgroup matrix strides along M
  MLX_MTL_CONST int TM_stride = 8 * WM;
  // Warp tile simdgroup matrix strides along M
  MLX_MTL_CONST int TN_stride = 8 * WN;

  // Leading dimensions of threadgroup A, B blocks
  MLX_MTL_CONST int lda_tgp = (transpose_a ? BM : BK) + tgp_padding_a;
  MLX_MTL_CONST int ldb_tgp = (transpose_b ? BK : BN) + tgp_padding_b;

  // Strides of A, B along reduction axis
  MLX_MTL_CONST short simd_stride_a =
      transpose_a ? TM_stride : TM_stride * lda_tgp;
  MLX_MTL_CONST short simd_stride_b =
      transpose_b ? TN_stride * ldb_tgp : TN_stride;

  // Jump between elements
  MLX_MTL_CONST short jump_a = transpose_a ? lda_tgp : 1;
  MLX_MTL_CONST short jump_b = transpose_b ? ldb_tgp : 1;

  // Offsets within threadgroup
  const int tm;
  const int tn;

  // Simdgroup matrices
  simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
  simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
  simdgroup_matrix<AccumType, 8, 8> results[TM * TN] = {
      simdgroup_matrix<AccumType, 8, 8>(0)};

  short sm;
  short sn;

  /* Constructor */
  METAL_FUNC Conv2DBlockMMA(
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : tm(8 * (simd_group_id / WN)), tn(8 * (simd_group_id % WN)) {
    short qid = simd_lane_id / 4;
    sm = (qid & 4) + (simd_lane_id / 2) % 4;
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
// Iterate over BK in blocks of 8
#pragma clang loop unroll(full)
    for (short kk = 0; kk < BK; kk += 8) {
      short2 offset_a =
          transpose_a ? short2(tm + sm, kk + sn) : short2(kk + sn, tm + sm);
      short2 offset_b =
          transpose_b ? short2(kk + sm, tn + sn) : short2(tn + sn, kk + sm);

      const threadgroup T* As__ = As + offset_a.y * lda_tgp + offset_a.x;
      const threadgroup T* Bs__ = Bs + offset_b.y * ldb_tgp + offset_b.x;

      simdgroup_barrier(mem_flags::mem_none);
// Load elements from threadgroup A as simdgroup matrices
#pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
        Asimd[i].thread_elements()[0] = static_cast<AccumType>(As__[0]);
        Asimd[i].thread_elements()[1] = static_cast<AccumType>(As__[jump_a]);
        As__ += simd_stride_a;
      }

      simdgroup_barrier(mem_flags::mem_none);
// Load elements from threadgroup B as simdgroup matrices
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        Bsimd[j].thread_elements()[0] = static_cast<AccumType>(Bs__[0]);
        Bsimd[j].thread_elements()[1] = static_cast<AccumType>(Bs__[jump_b]);
        Bs__ += simd_stride_b;
      }

      simdgroup_barrier(mem_flags::mem_none);
// Multiply and accumulate into result simdgroup matrices
#pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < TN; j++) {
          simdgroup_multiply_accumulate(
              results[i * TN + j], Asimd[i], Bsimd[j], results[i * TN + j]);
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device T* C, const int ldc) const {
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (int j = 0; j < TN; j++) {
        C[(i * TM_stride + sm + tm) * ldc + j * TN_stride + tn + sn] =
            Epilogue::apply(results[i * TN + j].thread_elements()[0]);
        C[(i * TM_stride + sm + tm) * ldc + j * TN_stride + tn + sn + 1] =
            Epilogue::apply(results[i * TN + j].thread_elements()[1]);
      }
    }
  }

  METAL_FUNC void
  store_result_safe(device T* C, const int ldc, short2 dst_tile_dims) const {
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      if (tm + i * TM_stride + sm < dst_tile_dims.y) {
#pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) {
          if (tn + j * TN_stride + sn < dst_tile_dims.x) {
            C[(tm + i * TM_stride + sm) * ldc + tn + j * TN_stride + sn] =
                Epilogue::apply(results[i * TN + j].thread_elements()[0]);
          }

          if (tn + j * TN_stride + sn + 1 < dst_tile_dims.x) {
            C[(tm + i * TM_stride + sm) * ldc + tn + j * TN_stride + sn + 1] =
                Epilogue::apply(results[i * TN + j].thread_elements()[1]);
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<T, AccumType>>
struct Conv2DImplicitGEMMKernel {
  MLX_MTL_CONST short tgp_padding_a = 16 / sizeof(T);
  MLX_MTL_CONST short tgp_padding_b = 16 / sizeof(T);
  MLX_MTL_CONST short tgp_mem_size_a =
      transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a);
  MLX_MTL_CONST short tgp_mem_size_b =
      transpose_b ? BN * (BK + tgp_padding_b) : BK * (BN + tgp_padding_b);
  MLX_MTL_CONST short tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;

  MLX_MTL_CONST short tgp_size = WM * WN * 32;
  MLX_MTL_CONST short vec_size = (BM == 64 && BN == 64) ? 8 : 4;

  using loader_a_t =
      Conv2DInputBlockLoader<T, BM, BN, BK, vec_size, tgp_size, tgp_padding_a>;
  using loader_b_t =
      Conv2DWeightBlockLoader<T, BM, BN, BK, vec_size, tgp_size, tgp_padding_b>;
  using mma_t = Conv2DBlockMMA<
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      tgp_padding_a,
      tgp_padding_b,
      AccumType,
      Epilogue>;

  /* Main kernel function */
  static METAL_FUNC void run(
      const device T* A [[buffer(0)]],
      const device T* B [[buffer(1)]],
      device T* C [[buffer(2)]],
      const constant MLXConvParams<2>& params [[buffer(3)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    const int c_row = tid.y * BM;
    const int c_col = tid.x * BN;
    const int K = params.wt_strides[0];
    const int N = params.O;

    B += c_col * K;
    C += c_row * N + c_col;

    // Prepare threadgroup memory for loading
    threadgroup T* As = tgp_memory;
    threadgroup T* Bs = tgp_memory + tgp_mem_size_a;

    // Prepare threadgroup loading operations
    loader_a_t loader_a(A, As, params, tid, lid, simd_gid, simd_lid);
    loader_b_t loader_b(B, Bs, params, tid, lid, simd_gid, simd_lid);

    // Prepare threadgroup mma operation
    mma_t mma_op(simd_gid, simd_lid);

    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Load elements into threadgroup
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Store results to device memory
    mma_op.store_result(C, N);
  }
};