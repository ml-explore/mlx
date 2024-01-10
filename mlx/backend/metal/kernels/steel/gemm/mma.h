// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = 8 * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = 8 * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

  // Strides of A, B along reduction axis
  STEEL_CONST short simd_stride_a = {
      transpose_a ? TM_stride : TM_stride * lda_tgp};
  STEEL_CONST short simd_stride_b = {
      transpose_b ? TN_stride * ldb_tgp : TN_stride};

  // Jump between elements
  STEEL_CONST short jump_a = {transpose_a ? lda_tgp : 1};
  STEEL_CONST short jump_b = {transpose_b ? ldb_tgp : 1};

  STEEL_CONST short tile_stride_a = {transpose_a ? 8 * lda_tgp : 8};
  STEEL_CONST short tile_stride_b = {transpose_b ? 8 : 8 * ldb_tgp};

  // Simdgroup matrices
  simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
  simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
  simdgroup_matrix<AccumType, 8, 8> results[TM * TN] = {
      simdgroup_matrix<AccumType, 8, 8>(0)};

  // Offsets within threadgroup
  const short tm;
  const short tn;

  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : tm(8 * (simd_group_id / WN)), tn(8 * (simd_group_id % WN)) {
    // Determine thread position in simdgroup matrix
    short qid = simd_lane_id / 4;
    sm = (qid & 4) + (simd_lane_id / 2) % 4;
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

    // Determine thread and simdgroup offset
    As_offset =
        transpose_a ? ((sn)*lda_tgp + (tm + sm)) : ((sn) + (tm + sm) * lda_tgp);
    Bs_offset =
        transpose_b ? ((tn + sn) * ldb_tgp + (sm)) : ((sm)*ldb_tgp + (tn + sn));
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of 8
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += 8) {
      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup A as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        Asimd[i].thread_elements()[0] =
            static_cast<AccumType>(As[i * simd_stride_a + 0]);
        Asimd[i].thread_elements()[1] =
            static_cast<AccumType>(As[i * simd_stride_a + jump_a]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup B as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        Bsimd[j].thread_elements()[0] =
            static_cast<AccumType>(Bs[j * simd_stride_b + 0]);
        Bsimd[j].thread_elements()[1] =
            static_cast<AccumType>(Bs[j * simd_stride_b + jump_b]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Multiply and accumulate into result simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          short j_serp = (i % 2) ? (TN - 1 - j) : j;

          simdgroup_multiply_accumulate(
              results[i * TN + j_serp],
              Asimd[i],
              Bsimd[j_serp],
              results[i * TN + j_serp]);
        }
      }

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* C, const int ldc) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset = (i * TM_stride) * ldc + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {Epilogue::apply(accum[0]), Epilogue::apply(accum[1])};

        // Write out C
        C[offset] = outs[0];
        C[offset + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void
  store_result_safe(device U* C, const int ldc, short2 dst_tile_dims) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn);
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset = (i * TM_stride) * ldc + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            C[offset] = Epilogue::apply(accum[0]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            C[offset + 1] = Epilogue::apply(accum[1]);
          }
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {
            epilogue_op.apply(accum[0], C[offset_c]),
            epilogue_op.apply(accum[1], C[offset_c + fdc])};

        // Write out D
        D[offset_d] = outs[0];
        D[offset_d + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            D[offset_d] = epilogue_op.apply(accum[0], C[offset_c]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            D[offset_d + 1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
          }
        }
      }
    }
  }
};

} // namespace steel
} // namespace mlx