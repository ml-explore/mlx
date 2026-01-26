// Copyright © 2025 Apple Inc.

using namespace mlx::steel;

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

// clang-format off
template <
    bool kAlignedM,
    bool kAlignedN,
    typename NAXTile_t,
    typename T>
void gemm_epilogue(
    thread NAXTile_t& Dtile,
    const device T* C,
    const constant GEMMParams* params,
    const constant GEMMAddMMParams* addmm_params,
    const short sgp_sm, 
    const short sgp_sn) { // clang-format on

  (void)params;

  constexpr short UM = NAXTile_t::kSubTileRows;
  constexpr short UN = NAXTile_t::kSubTileCols;
  using CSubTile = NAXSubTile<T, UM, UN>;

  using V = typename NAXTile_t::elem_type;

  constexpr short TM = NAXTile_t::kTileRows;
  constexpr short TN = NAXTile_t::kTileCols;
  constexpr short kElemsPerSubTile = NAXTile_t::kElemsPerSubTile;

  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < TN; nn++) {
      const short m = mm * UM;
      const short n = nn * UN;

      CSubTile CTile;

      if constexpr (kAlignedM && kAlignedN) {
        CTile.load(C, addmm_params->ldc, addmm_params->fdc, m, n);
      } else {
        CTile.load_safe(
            C, addmm_params->ldc, addmm_params->fdc, sgp_sm, sgp_sn, m, n);
      }

      auto delems = Dtile.subtile_at(mm, nn).elems();
      auto celems = CTile.elems();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerSubTile; i++) {
        if (do_axpby) {
          delems[i] = addmm_params->alpha * delems[i] +
              addmm_params->beta * static_cast<V>(celems[i]);
        } else {
          delems[i] += static_cast<V>(celems[i]);
        }
      }
    }
  }
}

// clang-format off
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
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2), function_constant(use_out_source)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
    const constant int64_t* batch_strides [[buffer(7), function_constant(has_batch)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on
  // Find block
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  // Exit early if out of bounds
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Adjust for batch
  if (has_batch) {
    const constant auto* A_bstrides = batch_strides;
    const constant auto* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    A += batch_offsets.x;
    B += batch_offsets.y;

    if (use_out_source) {
      const constant auto* C_bstrides = B_bstrides + params->batch_ndim;
      C += elem_to_loc(tid.z, batch_shape, C_bstrides, params->batch_ndim);
    }
  } else {
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;

    if (use_out_source) {
      C += addmm_params->batch_stride_c * tid.z;
    }
  }

  D += params->batch_stride_d * tid.z;

  // Prepare threadgroup memory
  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  D += c_row_long * params->ldd + c_col_long;

  if (use_out_source) {
    C += c_row_long * addmm_params->ldc + c_col_long * addmm_params->fdc;
  }

  constexpr short UM = 16;
  constexpr short UN = 32;
  constexpr short UK = 16;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / UM;
  constexpr short TN = SN / UN;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = align_M ? SM : min(SM, short(params->M - (c_row + tm)));
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);

  const short sgp_sn = align_N ? SN : min(SN, short(params->N - (c_col + tn)));
  const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

  A += transpose_a ? tm : (tm * params->lda);
  B += transpose_b ? (tn * params->ldb) : tn;
  D += tm * params->ldd + tn;

  if (use_out_source) {
    C += tm * addmm_params->ldc + tn * addmm_params->fdc;
  }

  using DSubTile = NAXSubTile<AccumType, UM, UN>;
  NAXTile<AccumType, TM, TN, DSubTile> Dtile;

  dispatch_bool(align_K, [&](auto kAlignedK) {
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
            UM,
            UN,
            UK,
            AccumType>(A, B, params, sgp_sm, sgp_sn);
        if (use_out_source) {
          gemm_epilogue<kAlignedM.value, kAlignedN.value>(
              Dtile, C, params, addmm_params, sgp_sm, sgp_sn);
        }
        if constexpr (kAlignedM && kAlignedN) {
          Dtile.store(D, int(params->ldd));
        } else {
          Dtile.store_safe(D, int(params->ldd), short2(sgp_sn, sgp_sm));
        }
      });
    });
  });
}
