// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename T, int kFragRows_ = 8, int kFragCols_ = 8>
struct BaseMMAFrag {
  static_assert(
      kFragRows_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
  static_assert(
      kFragCols_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");

  STEEL_CONST int kFragRows = kFragRows_;
  STEEL_CONST int kFragCols = kFragCols_;
  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread frag_type& A,
      thread frag_type& B,
      thread frag_type& C) {
    mat_type D_mat;
    mat_type A_mat;
    mat_type B_mat;
    mat_type C_mat;

    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread mat_type& A,
      thread mat_type& B,
      thread mat_type& C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  STEEL_CONST int kTileRows = kTileRows_;
  STEEL_CONST int kTileCols = kTileCols_;

  STEEL_CONST int kRows = kTileRows * kFragRows;
  STEEL_CONST int kCols = kTileCols * kFragCols;

  STEEL_CONST int kNumFrags = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MMATile() thread {}

  METAL_FUNC void clear() {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        frag_at(i, j) = frag_type(0);
      }
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC mat_type mat_at(const short i, const short j) {
    mat_type val_mat;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < kElemsPerFrag; ++ii) {
      val_mat.thread_elements()[ii] = frag_at(i, j)[ii];
    }
    return val_mat;
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        STEEL_PRAGMA_UNROLL
        for (int k = 0; k < kElemsPerFrag; ++k) {
          frag_at(i, j)[k] = static_cast<T>(
              src[(i * kFragRows) * w_x * str_x +
                  (j * kFragCols) * w_y * str_y + k * str_y]);
        }
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        STEEL_PRAGMA_UNROLL
        for (int k = 0; k < kElemsPerFrag; ++k) {
          dst[(i * kFragRows) * w_x * str_x + (j * kFragCols) * w_y * str_y +
              k * str_y] = static_cast<U>(frag_at(i, j)[k]);
        }
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void load(const device U* src, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        STEEL_PRAGMA_UNROLL
        for (int k = 0; k < kElemsPerFrag; ++k) {
          frag_at(i, j)[k] = static_cast<T>(
              src[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y + k]);
        }
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        STEEL_PRAGMA_UNROLL
        for (int k = 0; k < kElemsPerFrag; ++k) {
          dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y + k] =
              static_cast<U>(frag_at(i, j)[k]);
        }
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        STEEL_PRAGMA_UNROLL
        for (int k = 0; k < kElemsPerFrag; ++k) {
          int t_row = (i * kFragRows) * w_x;
          int t_col = (j * kFragCols) * w_y + k;
          if (t_row < src_tile_dims.y && t_col < src_tile_dims.x)
            frag_at(i, j)[k] = static_cast<T>(src[t_row * ld + t_col]);
        }
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        STEEL_PRAGMA_UNROLL
        for (int k = 0; k < kElemsPerFrag; ++k) {
          int t_row = (i * kFragRows) * w_x;
          int t_col = (j * kFragCols) * w_y + k;
          if (t_row < dst_tile_dims.y && t_col < dst_tile_dims.x)
            dst[t_row * ld + t_col] = static_cast<U>(frag_at(i, j)[k]);
        }
      }
    }
  }
};

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_matmad(
    thread MMATile<T, M, N>& D,
    thread MMATile<U, M, K>& A,
    thread MMATile<U, K, N>& B,
    thread MMATile<T, M, N>& C) {
  STEEL_PRAGMA_UNROLL
  for (int m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (int n = 0; n < N; ++n) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      STEEL_PRAGMA_UNROLL
      for (int k = 0; k < K; ++k) {
        MMATile<T, M, N>::MMAFrag_t::mma(
            D.frag_at(m, n_serp),
            A.frag_at(m, k),
            B.frag_at(k, n_serp),
            C.frag_at(m, n_serp));
      }
    }
  }
}

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
  // MMAFrag size
  STEEL_CONST short MMAFrag_size = 8;

  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = MMAFrag_size * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = MMAFrag_size * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

  // Threadgroup A strides
  STEEL_CONST int A_str_m = transpose_a ? 1 : lda_tgp; // M
  STEEL_CONST int A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  STEEL_CONST int B_str_k = transpose_b ? 1 : ldb_tgp; // K
  STEEL_CONST int B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  STEEL_CONST short tile_stride_a = MMAFrag_size * A_str_k;
  STEEL_CONST short tile_stride_b = MMAFrag_size * B_str_k;

  // Simdgroup matrices
  MMATile<AccumType, TM, 1> Atile;
  MMATile<AccumType, 1, TN> Btile;
  MMATile<AccumType, TM, TN> Ctile;

  // Offsets within threadgroup
  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    // Determine thread position in simdgroup matrix
    Ctile.clear();

    int tm = MMAFrag_size * (simd_group_id / WN);
    int tn = MMAFrag_size * (simd_group_id % WN);

    short2 simd_coord = decltype(Ctile)::MMAFrag_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    // Determine thread and simdgroup offset
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k; // M, K
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n; // K, N

    sm += tm;
    sn += tn;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of MMAFrag_size
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += MMAFrag_size) {
      simdgroup_barrier(mem_flags::mem_none);

      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);

      simdgroup_barrier(mem_flags::mem_none);

      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Ctile, Atile, Btile, Ctile);

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store<U, WM, WN>(D, ldd);
  }

  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }

  /* Apply epilogue */
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = epilogue_op.apply(Ctile.elems()[i]);
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Apply epilogue
        accum[0] = epilogue_op.apply(accum[0], C[offset_c]);
        accum[1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Read C
        U c_elems[2] = {0};

        if ((j * TN_stride + 1) < dst_tile_dims.x) {
          c_elems[0] = C[offset_c];
          c_elems[1] = C[offset_c + fdc];
        } else if ((j * TN_stride) < dst_tile_dims.x) {
          c_elems[0] = C[offset_c];
        }

        // Apply epilogue
        accum[0] = epilogue_op.apply(accum[0], c_elems[0]);
        accum[1] = epilogue_op.apply(accum[1], c_elems[1]);
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
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = Ctile.frag_at(i, j);
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
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = Ctile.frag_at(i, j);
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
