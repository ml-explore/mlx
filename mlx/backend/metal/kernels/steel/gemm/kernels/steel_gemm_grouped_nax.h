// Copyright © 2026 Apple Inc.

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

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

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

  const int s_row = c_row + tm;
  int group = 0;
  for (int last = num_groups - 1; group < last;) {
    int mid = (group + last + 1) / 2;
    if (offsets[mid] <= s_row) {
      group = mid;
    } else {
      last = mid - 1;
    }
  }

  short offset;
  short offset_next = 0;
  for (; offset_next < sgp_sm; group++) {
    offset = offset_next;
    offset_next = sgp_sm;
    if (group + 1 < num_groups) {
      offset_next =
          short(clamp(offsets[group + 1] - s_row, int(offset), int(sgp_sm)));
    }
    if (offset_next == offset) {
      continue;
    }
    threadgroup_barrier(mem_flags::mem_none);

    NAXTile<AccumType, TM, TN> Ctile;

    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
          auto do_gemm = gemm_loop<
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
              AccumType>;
          Ctile = do_gemm(
              A,
              B + group * params->batch_stride_b,
              params->lda,
              params->ldb,
              params->K,
              params->gemm_k_iterations_aligned,
              sgp_sm,
              sgp_sn);

          if constexpr (kAlignedN.value) {
            if (offset_next - offset == SM) {
              Ctile.store(C, int(params->ldd));
            } else {
              Ctile.store_slice(
                  C,
                  int(params->ldd),
                  short2(0, offset),
                  short2(SN, offset_next));
            }
          } else {
            Ctile.store_slice(
                C,
                int(params->ldd),
                short2(0, offset),
                short2(sgp_sn, offset_next));
          }
        });
      });
    });
  }
}
