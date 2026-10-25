// Copyright © 2024-26 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// Attention kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
constant bool save_lse [[function_constant(303)]];
constant int window_size [[function_constant(304)]];
constant bool has_window = window_size > 0;

// True if the key at position c is hidden from the query at position r
METAL_FUNC bool window_masked(int r, int c) {
  return (do_causal && r < c) ||
      (has_window &&
       ((r - c) >= window_size || (!do_causal && (c - r) >= window_size)));
}

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BV,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    device float* lse [[buffer(8), function_constant(save_lse)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (save_lse) {
    lse += tidl.z * params->H * params->qL + // Batch
        tidl.y * params->qL + // Head
        tidl.x * BQ; // Sequence
  }

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  // Blocks before this are ignored
  int kb_start = 0;

  // First block with no masked block edge
  int kb_win_lo = 0;
  // Last block with no masked block edge
  int kb_win_hi = params->NK;

  // Query positions covered by this block are [q_min, q_max)
  const int q_max = (tid.x + 1) * BQ + params->qL_off;
  const int q_min = max(0, int(tid.x) * BQ + params->qL_off);

  if (do_causal) {
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    kb_min_causal = (q_min / BK);
  } else if (has_window) {
    int k_max = q_max + window_size - 1;
    kb_lim = (k_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
  }

  if (has_window) {
    // The smallest k this block attends to is decided by the window
    int k_min = max(0, q_min - window_size + 1);
    kb_start = k_min / BK;

    // Move the K and V pointers to the first key block
    K += kb_start * BK * params->K_strides[2];
    V += kb_start * BK * params->V_strides[2];

    // All queries in the block [q_min, q_max) attend to
    // keys [common_k_min, common_k_max)
    auto common_k_min = max(0, q_max - window_size);
    auto common_k_max = q_min + window_size;
    kb_win_lo = (common_k_min + BK - 1) / BK;
    kb_win_hi = common_k_max / BK;
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BV + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BV + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;
  constexpr bool q_in_registers = (WN == 2) && (BK == 32 || sizeof(T) == 4);
  constexpr short q_tgp_mem_s = q_in_registers ? 1 : BQ * (BD + padQ);

  threadgroup T Q_smem[q_tgp_mem_s];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BV,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  using MMAInType = metal::conditional_t<WN == 2, T, AccumType>;
  using MMAFrag_in_t = BaseMMAFrag<MMAInType, kFragSize, kFragSize>;

  static_assert(WN == 1 || WN == 2, "WN must be 1 or 2");
  static_assert(
      BD % WN == 0 && BV % WN == 0,
      "The head dims must split evenly across WN");

  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;
  constexpr int TVh = BV / (kFragSize * WN);
  constexpr int BVh = BV / WN;

  static_assert(TQ == 1, "Check TQ");

  MMATile<MMAInType, TQ, 1, MMAFrag_in_t> Qtile;
  MMATile<MMAInType, 1, TK, MMAFrag_in_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<MMAInType, 1, 1, MMAFrag_in_t> Vtile;
  MMATile<AccumType, TQ, TVh, MMAFrag_acc_t> Otile;
  MMATile<MMAInType, 1, 1, MMAFrag_in_t> Qtiles[q_in_registers ? TDh : 1];

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;
  const short tm = kFragSize * TQ * row_group;

  const short Qs_offset = (tm + sm) * LDQ_tgp + d_half * BDh + sn;
  const short Ks_offset = d_half * BDh * LDK_tgp + sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + d_half * BVh + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (!q_in_registers) {
    if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
      loader_q.load_safe(short2(BD, params->qL_rem));
    } else {
      loader_q.load_unsafe();
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; dd++) {
      const device T* q_src = Q + (tm + sm) * params->Q_strides[2] +
          d_half * BDh + dd * kFragSize + sn;
      if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
        MMAFrag_in_t::load_safe(
            Qtiles[dd].frag_at(0, 0),
            q_src,
            params->Q_strides[2],
            Int<1>{},
            params->qL_rem - (tm + sm),
            BDh - (dd * kFragSize + sn),
            Int<0>{},
            Int<0>{});
      } else {
        MMAFrag_in_t::load(
            Qtiles[dd].frag_at(0, 0), q_src, params->Q_strides[2], Int<1>{});
      }
    }
  }

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  // Loop over KV seq length
  for (int kb = kb_start; kb < kb_lim; kb++) {
    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      if constexpr (!q_in_registers) {
        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + dd * Qs_tile_stride]);
      }
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      if constexpr (!q_in_registers) {
        tile_matmad(Stile, Qtile, Ktile, Stile);
      } else {
        tile_matmad(Stile, Qtiles[dd], Ktile, Stile);
      }
    }

    if constexpr (WN == 2) {
      constexpr int s_xchg_stride = decltype(Stile)::kElemsPerTile * 32;
      constexpr int s_xchg_size = WM * WN * s_xchg_stride;
      static_assert(
          s_xchg_size * sizeof(AccumType) <= tgp_mem_s * sizeof(T),
          "The score exchange must fit in KV_smem");

      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup AccumType* s_xchg =
          reinterpret_cast<threadgroup AccumType*>(KV_smem);
      threadgroup AccumType* slot =
          s_xchg + (row_group * WN + d_half) * s_xchg_stride;
      const short base = short(simd_lane_id) * decltype(Stile)::kElemsPerTile;

      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        slot[base + ii] = Stile.elems()[ii];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const threadgroup AccumType* peer =
          s_xchg + (row_group * WN + 1 - d_half) * s_xchg_stride;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] += peer[base + ii];
      }
    }

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out length sequence
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal or at the window edge
    const bool is_causal_masked = do_causal && (kb >= kb_min_causal);
    const bool is_win_lo_masked = has_window && (kb < kb_win_lo);
    const bool is_win_hi_masked = has_window && !do_causal && (kb >= kb_win_hi);

    if (is_causal_masked || is_win_lo_masked || is_win_hi_masked) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (has_window) {
              if (window_masked(row_pos, (col_pos + jj))) {
                Stile.frag_at(i, j)[jj] = neg_inf;
              }
            } else if (do_causal) {
              if (row_pos < (col_pos + jj)) {
                Stile.frag_at(i, j)[jj] = neg_inf;
              }
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += M_LOG2E_F * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (!align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BV, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TVh; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store the logsumexp for the backward pass
  if (save_lse && sn == 0) {
    using stile_t = decltype(Stile);
    const bool is_last_q = int(tid.x) == (params->NQ_aligned);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const short r = tm + sm + i * stile_t::kFragRows;
      if (align_Q || !is_last_q || r < params->qL_rem) {
        lse[r] = M_LN2_F * (max_score[i] + metal::log2(sum_score[i]));
      }
    }
  }

  // Store results
  O += (tm + sm) * params->O_strides[2] + d_half * BVh + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BVh - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Split KV attention kernel
///////////////////////////////////////////////////////////////////////////////

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_split(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant AttnParams* params [[buffer(6)]],
    const constant int& split_blocks [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) { // clang-format on

  constexpr short kFragSize = 8;
  constexpr int BV = BD;
  static_assert(WN == 1 || WN == 2, "A row tile spans one or two simdgroups");
  static_assert(BQ == WM * kFragSize, "Each simdgroup takes one row tile");

  const int qL = params->qL;
  const int gqa = params->gqa_factor;
  const int n_rows = gqa * qL;
  const int row_blocks = (n_rows + BQ - 1) / BQ;

  const int split = tid.x / row_blocks;
  const int blocks = tpg.x / row_blocks;
  const int tq = BQ * (tid.x % row_blocks);
  const int kv_head = tid.y;
  const int batch = tid.z;

  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BV + padV;
  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BV + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T KV_smem[tgp_mem_s];
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  const int kb_start = split * split_blocks;
  const int kb_end = min(kb_start + split_blocks, params->NK);

  K += batch * params->K_strides[0] + kv_head * params->K_strides[1] +
      int64_t(kb_start) * BK * params->K_strides[2];
  V += batch * params->V_strides[0] + kv_head * params->V_strides[1] +
      int64_t(kb_start) * BK * params->V_strides[2];

  constexpr short kThreads = WM * WN * 32;
  constexpr short kLoadThreads =
      kThreads >= 256 ? 256 : (kThreads >= 128 ? 128 : kThreads);
  const bool loads =
      kThreads == kLoadThreads || simd_group_id * 32 < kLoadThreads;

  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ kLoadThreads>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BV,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ kLoadThreads>;

  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  using MMAFrag_in_t = BaseMMAFrag<T, kFragSize, kFragSize>;

  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;
  constexpr int TVh = BV / (kFragSize * WN);
  constexpr int BVh = BV / WN;

  MMATile<T, 1, TK, MMAFrag_in_t> Ktile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Stile;
  MMATile<T, 1, 1, MMAFrag_in_t> Vtile;
  MMATile<AccumType, 1, TVh, MMAFrag_acc_t> Otile;
  MMATile<T, 1, 1, MMAFrag_in_t> Qtiles[TDh];

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;
  const short tm = kFragSize * row_group;

  const short Ks_offset = d_half * BDh * LDK_tgp + sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + d_half * BVh + sn;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  const int row = tq + tm + sm;
  const int r = row % qL;
  const int q_pos = params->qL_off + r;
  const bool row_valid = row < n_rows;
  const device T* q_src = Q + batch * params->Q_strides[0] +
      (kv_head * gqa + row / qL) * params->Q_strides[1] +
      r * params->Q_strides[2] + d_half * BDh + sn;
  STEEL_PRAGMA_UNROLL
  for (short dd = 0; dd < TDh; dd++) {
    if (row_valid) {
      MMAFrag_in_t::load(
          Qtiles[dd].frag_at(0, 0),
          q_src + dd * kFragSize,
          params->Q_strides[2],
          Int<1>{});
    } else {
      Qtiles[dd].frag_at(0, 0) = typename MMAFrag_in_t::frag_type(0);
    }
  }

  AccumType max_score = Limits<AccumType>::finite_min;
  AccumType sum_score = 0;

  for (int kb = kb_start; kb < kb_end; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (loads) {
      if (!align_K && kb == (params->NK_aligned)) {
        loader_k.load_safe(short2(BD, params->kL_rem));
      } else {
        loader_k.load_unsafe();
      }
    }

    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtiles[dd], Ktile, Stile);
    }

    if constexpr (WN == 2) {
      constexpr int s_xchg_stride = decltype(Stile)::kElemsPerTile * 32;
      constexpr int s_xchg_size = WM * WN * s_xchg_stride;
      static_assert(
          s_xchg_size * sizeof(AccumType) <= tgp_mem_s * sizeof(T),
          "The score exchange must fit in KV_smem");

      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup AccumType* s_xchg =
          reinterpret_cast<threadgroup AccumType*>(KV_smem);
      threadgroup AccumType* slot =
          s_xchg + (row_group * WN + d_half) * s_xchg_stride;
      const short base = short(simd_lane_id) * decltype(Stile)::kElemsPerTile;

      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        slot[base + ii] = Stile.elems()[ii];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const threadgroup AccumType* peer =
          s_xchg + (row_group * WN + 1 - d_half) * s_xchg_stride;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] += peer[base + ii];
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    using stile_t = decltype(Stile);
    constexpr auto neg_inf = Limits<AccumType>::finite_min;

    if (!align_K && kb == (params->NK_aligned)) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < stile_t::kTileCols; j++) {
        short col_pos = sn + (j * stile_t::kFragCols);
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
          if ((col_pos + jj) >= params->kL_rem) {
            Stile.frag_at(0, j)[jj] = neg_inf;
          }
        }
      }
    }

    if (do_causal && (kb + 1) * BK > params->qL_off + 1) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < stile_t::kTileCols; j++) {
        const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
          if (q_pos < (col_pos + jj)) {
            Stile.frag_at(0, j)[jj] = neg_inf;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (loads) {
      if (!align_K && kb == (params->NK_aligned)) {
        loader_v.load_safe(short2(BV, params->kL_rem));
      } else {
        loader_v.load_unsafe();
      }
    }

    AccumType new_max = max_score;
    Stile.template row_reduce<MaxOp>(&new_max);
    Stile.template row_bin_op<ExpSubOp>(&new_max);
    AccumType factor = fast::exp2(max_score - new_max);
    max_score = new_max;

    AccumType sum_score_tmp = 0;
    Stile.template row_reduce<SumOp>(&sum_score_tmp);
    sum_score = sum_score * factor + sum_score_tmp;

    Otile.template row_bin_op<MulOp>(&factor);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TVh; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        simdgroup_barrier(mem_flags::mem_none);
        Vtile.template load<T, 1, 1, LDV_tgp, 1>(
            &Vs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
        simdgroup_barrier(mem_flags::mem_none);
        MMAFrag_acc_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }

    loader_k.next();
    loader_v.next();
  }

  AccumType keep = (max_score == Limits<AccumType>::finite_min) ? 0 : 1;
  sum_score *= keep;
  Otile.template row_bin_op<MulOp>(&keep);

  if (!row_valid) {
    return;
  }

  const int64_t idx = (int64_t(batch) * params->H + kv_head * gqa) * qL + row;
  if (d_half == 0 && sn == 0) {
    sums[idx * blocks + split] = sum_score;
    maxs[idx * blocks + split] = max_score * M_LN2_F;
  }

  O += (idx * blocks + split) * BD + d_half * BVh + sn;
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TVh; id++) {
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 2; jj++) {
      O[id * kFragSize + jj] = Otile.frag_at(0, id)[jj];
    }
  }
}