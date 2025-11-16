// Copyright © 2024 Apple Inc.

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
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
gather_mm_rhs_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* rhs_indices [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short UM = 16;
  constexpr short UN = 32;
  constexpr short UK = 16;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / UM;
  constexpr short TN = SN / UN;

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  // Find the block in A, B, C
  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;
  rhs_indices += c_row;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = align_M ? SM : min(SM, short(params->M - (c_row + tm)));
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);

  const short sgp_sn = align_N ? SN : min(SN, short(params->N - (c_col + tn)));
  const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

  A += transpose_a ? tm : (tm * params->lda);
  B += transpose_b ? (tn * params->ldb) : tn;
  C += tm * params->ldd + tn;
  rhs_indices += tm;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = rhs_indices[0];
  short offset_next = 0;
  int n = 0;
  while (n < sgp_sm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = sgp_sm;
    for (; n < sgp_sm; n++) {
      if (rhs_indices[n] != index) {
        offset_next = n;
        index_next = rhs_indices[n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    using DSubTile = NAXSubTile<AccumType, UM, UN>;
    NAXTile<AccumType, TM, TN, DSubTile> Ctile;

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
              UM,
              UN,
              UK,
              AccumType>;
          Ctile = do_gemm(
              A, B + index * params->batch_stride_b, params, sgp_sm, sgp_sn);

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
