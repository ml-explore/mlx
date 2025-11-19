// Copyright © 2024-25 Apple Inc.

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

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
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

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

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short UQ = 16;
  constexpr short UD = 32;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * UQ) && BQ % (kNWarps * UQ) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * UQ);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / UD;

  static_assert(TQ == 1, "Check TQ");

  using OSubTile = NAXSubTile<AccumType, UQ, UD>;
  NAXTile<AccumType, TQ, TD, OSubTile> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = OSubTile::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = UQ * TQ * simd_group_id;

  Q += (tm + sm) * int(params->Q_strides[2]) + sn;
  K += sm * int(params->K_strides[2]) + sn;
  V += sm * int(params->V_strides[2]) + sn;

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Otile)::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

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

  int kb_lim = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - (tm + sm);
  const short lim_rows_k = params->kL_rem - sm;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    constexpr short UDs = 16;
    constexpr short UKs = 32;

    constexpr short TDs = BD / UDs;
    constexpr short TKs = BK / UKs;

    using SSubTile = NAXSubTile<AccumType, UQ, UKs>;
    using QSubTile = NAXSubTile<T, UQ, UDs>;
    using KSubTile = NAXSubTile<T, UKs, UDs>;

    NAXTile<AccumType, TQ, TKs, SSubTile> Stile;

    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TKs; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDs; id++) {
          NAXTile<T, 1, 1, QSubTile> Qtile;
          NAXTile<T, 1, 1, KSubTile> Ktile;

          const int Q_load_off = iq * UQ * int(params->Q_strides[2]) + id * UDs;
          const int K_load_off =
              ik * UKs * int(params->K_strides[2]) + id * UDs;

          if (!align_Q && is_last_q) {
            // Qtile.load_rows(
            //     Q + Q_load_off,
            //     int(params->Q_strides[2]),
            //     lim_rows_q - iq * UQ);
            Qtile.load_safe(
                Q + Q_load_off,
                int(params->Q_strides[2]),
                short2(BD, lim_rows_q - iq * UQ));
          } else {
            Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));
          }

          if (!align_K && is_last_k) {
            // Ktile.load_rows(
            //     K + K_load_off,
            //     int(params->K_strides[2]),
            //     lim_rows_k - ik * UKs);
            Ktile.load_safe(
                K + K_load_off,
                int(params->K_strides[2]),
                short2(BD, lim_rows_k - ik * UKs));
          } else {
            Ktile.load(K + K_load_off, int(params->K_strides[2]));
          }

          subtile_matmad_nax(
              Stile.subtile_at(iq, ik),
              Qtile.subtile_at(0, 0),
              metal::false_type{},
              Ktile.subtile_at(0, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Scale and Retile S
    constexpr short UK = 16;
    constexpr short TK = BK / UK;
    using PSubTile = NAXSubTile<AccumType, UQ, UK>;

    NAXTile<AccumType, TQ, TK, PSubTile> Ptile;

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Ptile.elems()[ii] = Stile.elems()[ii];
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = sn + ik * UK;

          thread auto& fg = Ptile.subtile_at(iq, ik).frag_at(0, 0);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < PSubTile::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < PSubTile::kFragThrCols; jj++) {
              const auto loc = ii * PSubTile::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) >= params->kL_rem) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= (kb_lim - ((BQ + BK - 1) / BK) - int(!align_K))) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short row_pos = base_row + iq * UQ;
          const short col_pos = base_col + ik * UK;

          thread auto& fg = Ptile.subtile_at(iq, ik).frag_at(0, 0);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < PSubTile::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < PSubTile::kFragThrCols; jj++) {
              const auto r = row_pos + ii * PSubTile::kFragRowsJump + sm;
              const auto c = col_pos + jj + sn;
              const auto loc = ii * PSubTile::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using MSubTile = NAXSubTile<melem_t, UQ, UK>;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short row_pos = base_row + iq * UQ + sm;
          const short col_pos = base_col + ik * UK + sn;

          MSubTile mfrag;
          mfrag.load_safe(
              mask,
              int(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          thread auto& fg = Ptile.subtile_at(iq, ik).frag_at(0, 0);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < MSubTile::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag.elems()[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag.elems()[jj]);
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Ptile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Ptile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Ptile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        if constexpr (BD == 128) {
          if (id == 2) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          using VSubTile = NAXSubTile<T, UK, UD>;
          NAXTile<T, 1, 1, VSubTile> Vtile;

          const int V_load_off = ik * UK * int(params->V_strides[2]) + id * UD;

          if (!align_K && is_last_k) {
            // Vtile.load_rows(
            //     V + V_load_off,
            //     int(params->V_strides[2]),
            //     lim_rows_k - ik * UK);
            Vtile.load_safe(
                V + V_load_off,
                int(params->V_strides[2]),
                short2(BD, lim_rows_k - ik * UK));
          } else {
            Vtile.load(V + V_load_off, int(params->V_strides[2]));
          }

          subtile_matmad_nax(
              Otile.subtile_at(iq, id),
              Ptile.subtile_at(iq, ik),
              metal::bool_constant<false>{},
              Vtile.subtile_at(0, 0),
              metal::bool_constant<false>{});
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = (1.f / sum_score[i]);
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += (tm + sm) * int(params->O_strides[2]) + sn;

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    // Otile.store_rows(O, params->O_strides[2], lim_rows_q);
    Otile.store_safe(O, params->O_strides[2], short2(BD, lim_rows_q));
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}
