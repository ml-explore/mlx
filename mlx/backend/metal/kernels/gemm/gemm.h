// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#define MLX_MTL_CONST static constant constexpr const

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BROWS,
    int BCOLS,
    int BK,
    int vec_size,
    int tgp_size,
    bool transpose,
    bool ldK,
    int tgp_padding = 0>
struct BlockLoader {
  // Destination dimensions
  MLX_MTL_CONST int dst_fd = transpose ? BCOLS : BROWS;
  MLX_MTL_CONST int dst_ld = (transpose ? BROWS : BCOLS) + tgp_padding;
  MLX_MTL_CONST int n_vecs = (transpose ? BROWS : BCOLS) / vec_size;

  // Stride along block row within the block
  MLX_MTL_CONST int bstride = tgp_size / n_vecs;

  // Leading dimension for src
  const int src_ld;
  // Stride along reduction axis between blocks
  const int tstride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tstride(
            BK * ((int)(transpose ^ !ldK) * src_ld + (int)(transpose ^ ldK))),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / n_vecs),
        bj(vec_size * (thread_idx % n_vecs)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
#pragma clang loop unroll(full)
    for (short i = 0; i < dst_fd; i += bstride) {
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = transpose ? src_tile_dim.yx : src_tile_dim.xy;

    // Iterate over rows of block
#pragma clang loop unroll(full)
    for (short i = 0; i < dst_fd; i += bstride) {
      // Row is in bounds, we check against column
      if ((bi + i) < src_tile_dim.y) {
        // Use fast thread memory for bound checks
        short tmp_idx[vec_size];
        T tmp_val[vec_size];

        // Make sure tmp_idx only contains valid indices
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          tmp_idx[j] = bj + j < src_tile_dim.x ? j : 0;
        }

        // Read all valid indcies into tmp_val
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          tmp_val[j] = src[i * src_ld + tmp_idx[j]];
        }

        // Zero out uneeded values
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          tmp_val[j] = bj + j < src_tile_dim.x ? tmp_val[j] : T(0);
        }

        // Copy values to threadgroup memory
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = tmp_val[j];
        }
      }

      // Row is out of bounds, we just fill tgp memory with zeros
      else {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tstride;
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
struct BlockMMA {
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
  METAL_FUNC BlockMMA(
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
// Multiply and accumulate into resulr simdgroup matrices
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
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<T, AccumType>>
struct GEMMKernel {
  MLX_MTL_CONST short tgp_padding_a = 16 / sizeof(T);
  MLX_MTL_CONST short tgp_padding_b = 16 / sizeof(T);
  MLX_MTL_CONST short tgp_mem_size_a =
      transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a);
  MLX_MTL_CONST short tgp_mem_size_b =
      transpose_b ? BN * (BK + tgp_padding_b) : BK * (BN + tgp_padding_b);
  MLX_MTL_CONST short tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;

  MLX_MTL_CONST short tgp_size = WM * WN * 32;
  MLX_MTL_CONST short vec_size = (BM == 64 && BN == 64) ? 8 : 4;

  using loader_a_t = BlockLoader<
      T,
      BM,
      BK,
      BK,
      vec_size,
      tgp_size,
      transpose_a,
      true,
      tgp_padding_a>;
  using loader_b_t = BlockLoader<
      T,
      BK,
      BN,
      BK,
      vec_size,
      tgp_size,
      transpose_b,
      false,
      tgp_padding_b>;
  using mma_t = BlockMMA<
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
      const constant int& M [[buffer(3)]],
      const constant int& N [[buffer(4)]],
      const constant int& K [[buffer(5)]],
      const constant int& batch_stride_a [[buffer(6)]],
      const constant int& batch_stride_b [[buffer(7)]],
      const constant int& batch_stride_c [[buffer(8)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    // Pacifying compiler
    (void)lid;

    // Adjust for batch
    A += batch_stride_a * tid.z;
    B += batch_stride_b * tid.z;
    C += batch_stride_c * tid.z;

    // Adjust for transpose
    const int lda_dev = transpose_a ? M : K;
    const int ldb_dev = transpose_b ? K : N;

    // Find block in A, B, C
    const int c_row = tid.y * BM;
    const int c_col = tid.x * BN;

    A += transpose_a ? c_row : c_row * K;
    B += transpose_b ? c_col * K : c_col;
    C += c_row * N + c_col;

    // Prepare threadgroup memory for loading
    threadgroup T* As = tgp_memory;
    threadgroup T* Bs = tgp_memory + tgp_mem_size_a;

    // Prepare threadgroup loading operations
    loader_a_t loader_a(A, lda_dev, As, simd_group_id, simd_lane_id);
    loader_b_t loader_b(B, ldb_dev, Bs, simd_group_id, simd_lane_id);

    // Prepare threadgroup mma operation
    mma_t mma_op(simd_group_id, simd_lane_id);

    ///////////////////////////////////////////////////////////////////////////////
    // MNK aligned loop
    if (MN_aligned && K_aligned) {
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
      return;

    }
    ///////////////////////////////////////////////////////////////////////////////
    // MN aligned, K unaligned loop
    else if (MN_aligned && !K_aligned) {
      // Main loop
      int k = 0;
      for (; k + BK <= K; k += BK) {
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

      // Loop tail
      threadgroup_barrier(mem_flags::mem_threadgroup);

      loader_a.load_safe(short2(K - k, BM));
      loader_b.load_safe(short2(BN, K - k));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_op.mma(As, Bs);

      // Store results to device memory
      mma_op.store_result(C, N);
      return;

    }
    ///////////////////////////////////////////////////////////////////////////////
    // MNK unaligned loop
    else { // Loop over K - unaligned case

      short2 src_tile_dims(min(BN, N - c_col), min(BM, M - c_row));

      if (src_tile_dims.y == BM && src_tile_dims.x == BN) {
        int k = 0;
        for (; k + BK <= K; k += BK) {
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

        if (k < K) {
          loader_a.load_safe(short2(K - k, BM));
          loader_b.load_safe(short2(BN, K - k));

          threadgroup_barrier(mem_flags::mem_threadgroup);

          mma_op.mma(As, Bs);
        }

        mma_op.store_result(C, N);
        return;

      } else {
        int k = 0;
        for (; k + BK <= K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          // Load elements into threadgroup
          loader_a.load_safe(short2(BK, src_tile_dims.y));
          loader_b.load_safe(short2(src_tile_dims.x, BK));

          threadgroup_barrier(mem_flags::mem_threadgroup);

          // Multiply and accumulate threadgroup elements
          mma_op.mma(As, Bs);

          // Prepare for next iteration
          loader_a.next();
          loader_b.next();
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (k < K) {
          loader_a.load_safe(short2(K - k, src_tile_dims.y));
          loader_b.load_safe(short2(src_tile_dims.x, K - k));

          threadgroup_barrier(mem_flags::mem_threadgroup);

          mma_op.mma(As, Bs);
        }

        threadgroup_barrier(mem_flags::mem_none);
        mma_op.store_result_safe(C, N, src_tile_dims);

        return;
      }
    }
  }
};