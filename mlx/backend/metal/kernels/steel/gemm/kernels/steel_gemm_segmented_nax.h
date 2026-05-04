// Copyright © 2026 Apple Inc.

using namespace mlx::steel;

constant bool segments_contiguous [[function_constant(199)]];
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];

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
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void segmented_mm_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* segments [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  const int tid_m = (BK > 64) ? tid.y : tid.z;
  const int tid_n = (BK > 64) ? tid.x : tid.y;
  const int tid_s = (BK > 64) ? tid.z : tid.x;

  const int c_row = tid_m * BM;
  const int c_col = tid_n * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  if (params->tiles_n <= static_cast<int>(tid_n) ||
      params->tiles_m <= static_cast<int>(tid_m)) {
    return;
  }

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

  uint32_t k_start, k_end;
  if (segments_contiguous) {
    k_start = segments[2 * tid_s];
    k_end = segments[2 * tid_s + 1];
  } else {
    k_start = segments[tid_s];
    k_end = segments[tid_s + 1];
  }
  A += transpose_a ? k_start * params->lda : k_start;
  B += transpose_b ? k_start : k_start * params->ldb;
  C += tid_s * params->batch_stride_d;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const int sgp_sm_int =
      align_M ? int(SM) : min(int(SM), params->M - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);

  const int sgp_sn_int =
      align_N ? int(SN) : min(int(SN), params->N - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);
  const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

  A += transpose_a ? tm : (tm * params->lda);
  B += transpose_b ? (tn * params->ldb) : tn;
  C += tm * params->ldd + tn;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const int segment_k_size = k_end - k_start;
  const int segment_k_iters = segment_k_size / BK;
  const bool segment_k_aligned = (segment_k_size % BK) == 0;

  dispatch_bool(segment_k_aligned, [&](auto kAlignedK) {
    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
        Dtile = gemm_loop<
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
            segment_k_size,
            segment_k_iters,
            sgp_sm,
            sgp_sn);
      });
    });
  });

  dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
      if constexpr (kAlignedM && kAlignedN) {
        Dtile.store(C, int(params->ldd));
      } else {
        Dtile.store_safe(C, int(params->ldd), short2(sgp_sn, sgp_sm));
      }
    });
  });
}
