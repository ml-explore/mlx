// Copyright © 2024 Apple Inc.

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

constant bool do_gather [[function_constant(300)]];

constant bool gather_bias = do_gather && use_out_source;

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
    return fast::exp(x - y);
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
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Seqeunce

  K += tidl.z * params->K_strides[0] + // Batch
      tidl.y * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      tidl.y * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Seqeunce

  constexpr int padQ = 0; // 16 / sizeof(T);
  constexpr int padK = 0; // 16 / sizeof(T);
  constexpr int padV = 0; // 16 / sizeof(T);

  // using QBlockSrcShape = CShape<BQ, BD>;
  // using KBlockSrcShape = CShape<BK, BD>;
  // using VBlockSrcShape = CShape<BK, BD>;

  constexpr int LDQ_tgp = BD + padQ;
  constexpr int LDK_tgp = BK + padK;
  constexpr int LDV_tgp = BD + padV;

  // using QBlockDstStrides = CShape<LDQ_tgp, 1>;
  // using KBlockDstStrides = CShape<1, LDK_tgp>;
  // using QBlockDstStrides = CShape<LDV_tgp, 1>;

  threadgroup T Qs[BQ * (BD + padQ)];
  threadgroup T Ks[(BK + padK) * BD];
  threadgroup T Vs[BK * (BD + padV)];

  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

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
      /* short BCOLS = */ BD,
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

  TransformScale<T> ts(static_cast<T>(params->scale));

  // MMAFrag size
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, TK, TD, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  short sm = simd_coord.y;
  short sn = simd_coord.x;
  short tm = kFragSize * TQ * simd_group_id;

  short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  short Ks_offset = sm * LDK_tgp + sn;
  short Vs_offset = sm * LDV_tgp + sn;

  constexpr int Qs_tile_stride = kFragSize;
  constexpr int Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  loader_q.load_unsafe();
  loader_q.apply_inplace_op(ts);

  constexpr int kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::min;
  }

  for (int kb = 0; kb < params->NK; kb++) {
    // Load Q and K blocks and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_k.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do S = Q @ K.T
    Stile.clear();

    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    simdgroup_barrier(mem_flags::mem_none);

    // Load V blocks
    loader_v.load_unsafe();

    // Do softmax

    // Row max
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp(max_score[i] - new_max[i]);
    }
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

    // Do O = S @ V
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Vtile.template load<T, 1, 1, LDV_tgp, 1>(&Vs[Vs_offset]);

    simdgroup_barrier(mem_flags::mem_none);

    tile_matmad(Otile, Stile, Vtile, Otile);

    // Prepare for next iteration
    // loader_q.next();
    loader_k.next();
    loader_v.next();
  }

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  O += (tm + sm) * params->O_strides[2] + sn;
  Otile.template store<T, 1, 1>(O, params->O_strides[2]);
}
