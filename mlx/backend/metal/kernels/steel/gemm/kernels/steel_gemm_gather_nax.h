// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_utils.h"

using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
gather_mm_rhs_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device int32_t* offsets [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& num_groups [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;

  int c_row;
  int group;
  short tgp_bm;
  if (params->tiles_n <= static_cast<int>(tid.x) ||
      !gather_mm_row_tile<BM>(
          offsets,
          num_groups,
          params->M,
          tid.y,
          simd_lane_id,
          c_row,
          group,
          tgp_bm)) {
    return;
  }

  const int c_col = tid.x * BN;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = short(clamp(tgp_bm - tm, 0, int(SM)));
  const short sgp_sn =
      align_N ? SN : short(clamp(params->N - (c_col + tn), 0, int(SN)));

  const size_t c_row_long = size_t(c_row + tm);
  const size_t c_col_long = size_t(c_col + tn);
  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += group * params->batch_stride_b;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

  dispatch_bool(align_K, [&](auto kAlignedK) {
    dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
      dispatch_bool(sgp_sn == SN, [&](auto kAlignedN) {
        NAXTile<AccumType, TM, TN> Ctile = gemm_loop<
            T,
            SM,
            SN,
            SK,
            BK,
            transpose_a,
            transpose_b,
            kAlignedM.value,
            kAlignedN.value,
            kAlignedK.value,
            AccumType>(
            A,
            B,
            params->lda,
            params->ldb,
            params->K,
            params->gemm_k_iterations_aligned,
            sgp_sm,
            sgp_sn);

        if constexpr (kAlignedM.value && kAlignedN.value) {
          Ctile.store(C, int(params->ldd));
        } else {
          Ctile.store_safe(C, int(params->ldd), short2(sgp_sn, sgp_sm));
        }
      });
    });
  });
}
