// Copyright © 2026 Apple Inc.

using namespace mlx::steel;

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
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void grouped_mm_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device int32_t* offsets [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& num_groups [[buffer(5)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;

  const int group = tid.y;
  if (params->tiles_n <= static_cast<int>(tid.x) || group >= num_groups) {
    return;
  }

  const int c_col = tid.x * BN;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short tgp_bn = align_N ? BN : short(min(BN, params->N - c_col));
  const short sgp_sn = short(clamp(int(tgp_bn) - tn, 0, int(SN)));
  const int group_end = group + 1 < num_groups ? offsets[group + 1] : params->M;

  const size_t c_col_long = size_t(c_col + tn);
  B += group * params->batch_stride_b;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_col_long;

  for (int c_row = offsets[group]; c_row < group_end; c_row += BM) {
    const short tgp_bm = short(min(BM, group_end - c_row));
    const short sgp_sm = short(clamp(int(tgp_bm) - tm, 0, int(SM)));
    const size_t c_row_long = size_t(c_row + tm);
    const device T* A_tile =
        A + (transpose_a ? c_row_long : c_row_long * params->lda);
    device T* C_tile = C + c_row_long * params->ldd;

    threadgroup_barrier(mem_flags::mem_none);

    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(tgp_bm == BM, [&](auto kAlignedM) {
        dispatch_bool(align_N || tgp_bn == BN, [&](auto kAlignedN) {
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
              A_tile,
              B,
              params->lda,
              params->ldb,
              params->K,
              params->gemm_k_iterations_aligned,
              sgp_sm,
              sgp_sn);

          if constexpr (kAlignedM.value && kAlignedN.value) {
            Ctile.store(C_tile, int(params->ldd));
          } else {
            Ctile.store_safe(C_tile, int(params->ldd), short2(sgp_sn, sgp_sm));
          }
        });
      });
    });
  }
}
