// Copyright © 2026 Apple Inc.

using namespace mlx::steel;

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

// Each simdgroup computes one (16 * UM) x (16 * UN) block of D. It keeps a K
// chunk of A in registers and MPP reads the B chunk from device memory.
// KS groups of SM x SN simdgroups split K.
// clang-format off
template <
    typename T,
    int UM,
    int UN,
    int SM,
    int SN,
    int KS,
    bool transpose_b>
[[kernel, max_total_threads_per_threadgroup(SM * SN * KS * 32)]] void gemm_thin_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2), function_constant(use_out_source)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using tensor_t = metal::tensor<device T, extents_t, metal::tensor_inline>;

  constexpr int TSM = 16 * UM;
  constexpr int TSN = 16 * UN;
  constexpr int BK = 64;

  const int M = params->M;
  const int N = params->N;
  const int K = params->K;
  const int lda = params->lda;
  const int ldb = params->ldb;

  // Threadgroups go along M first, so groups that run together share B.
  const int tgp = tid.x;
  const int tile_m = tgp % params->tiles_m;
  const int tile_n = tgp / params->tiles_m;

  const int k_group = simd_group_id / (SM * SN);
  const int sg = simd_group_id % (SM * SN);
  int m_off = (tile_m * SM + sg % SM) * TSM;
  int n_off = (tile_n * SN + sg / SM) * TSN;

  // A block at the edge moves back inside if the matrix is large enough. It
  // then writes again some outputs of a neighbor block, with equal values.
  const bool active = (align_M || m_off < M) && (align_N || n_off < N);
  if (!align_M && m_off + TSM > M && M >= TSM) {
    m_off = M - TSM;
  }
  if (!align_N && n_off + TSN > N && N >= TSN) {
    n_off = N - TSN;
  }
  const bool inside =
      (align_M || m_off + TSM <= M) && (align_N || n_off + TSN <= N);
  const int rows = active ? M - m_off : 0;
  const int cols = active ? N - n_off : 0;

  A += int64_t(m_off) * lda;
  B += transpose_b ? int64_t(n_off) * ldb : int64_t(n_off);
  D += int64_t(m_off) * params->ldd + n_off;
  if (use_out_source) {
    C +=
        int64_t(m_off) * addmm_params->ldc + int64_t(n_off) * addmm_params->fdc;
  }

  tensor_t tA((device T*)A, extents_t(K, rows), strides_t{1, lda});
  tensor_t tB(
      (device T*)B,
      transpose_b ? extents_t(K, cols) : extents_t(cols, K),
      strides_t{1, ldb});

  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) {
    if constexpr (transpose_b) {
      return tB.template slice<BK, TSN>(k, 0);
    } else {
      return tB.template slice<TSN, BK>(0, k);
    }
  };
  auto b_tail = [&](int k) {
    if constexpr (transpose_b) {
      return tB.template slice<metal::dynamic_extent, TSN>(k, 0);
    } else {
      return tB.template slice<TSN, metal::dynamic_extent>(0, k);
    }
  };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      transpose_b,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  // The accumulator layout depends only on the element types, so the masked
  // runs below can also use it.
  auto Dacc = op.template get_destination_cooperative_tensor<
      a_chunk_t,
      b_chunk_t,
      float>();
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op.template get_left_input_cooperative_tensor<T, T, float>();

  const int k_iters = params->gemm_k_iterations_aligned;
  for (int it = 0; it < k_iters; it++) {
    // Keep the simdgroups near each other in K, so that their shared loads
    // hit in cache.
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + k_group) * BK;
    if (!active || k >= K) {
      continue;
    }
    if (inside && (align_K || k + BK <= K)) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op.run(Areg, b, Dacc);
    } else if (inside) {
      auto a = tA.template slice<metal::dynamic_extent, TSM>(k, 0);
      auto b = b_tail(k);
      op.run(a, b, Dacc);
    } else {
      // The views stop at the matrix edge and MPP masks the rest.
      const int kk = min(BK, K - k);
      tensor_t vA((device T*)A + k, extents_t(kk, rows), strides_t{1, lda});
      tensor_t vB(
          (device T*)B + (transpose_b ? int64_t(k) : int64_t(k) * ldb),
          transpose_b ? extents_t(kk, cols) : extents_t(cols, kk),
          strides_t{1, ldb});
      op.run(vA, vB, Dacc);
    }
  }

  if constexpr (KS > 1) {
    // K groups 1 to KS - 1 store their partial sums and group 0 adds them.
    using tg_tensor_t =
        metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;
    constexpr int kBlockSize = TSM * TSN;
    static_assert(
        (KS - 1) * SM * SN * kBlockSize <= 4096,
        "Partial sums need more than 16 KB of threadgroup memory");
    threadgroup float partials[(KS - 1) * SM * SN * kBlockSize];

    if (active && k_group > 0) {
      tg_tensor_t tP(
          partials + ((k_group - 1) * SM * SN + sg) * kBlockSize,
          extents_t(TSN, TSM),
          strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && k_group == 0) {
      for (int g = 0; g < KS - 1; g++) {
        tg_tensor_t tP(
            partials + (g * SM * SN + sg) * kBlockSize,
            extents_t(TSN, TSM),
            strides_t{1, TSN});
        auto part = op.template get_destination_cooperative_tensor<
            a_chunk_t,
            b_chunk_t,
            float>();
        part.load(tP);
        STEEL_PRAGMA_UNROLL
        for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
          Dacc[i] += part[i];
        }
      }
    }
  }
  if (k_group != 0 || !active) {
    return;
  }

  auto Dout =
      op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, T>();
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
    float d = Dacc[i];
    if (use_out_source) {
      auto idx = Dacc.get_multidimensional_index(i);
      const int r = idx[1];
      const int c = idx[0];
      if (inside || (r < rows && c < cols)) {
        float c_val = static_cast<float>(
            C[int64_t(r) * addmm_params->ldc + int64_t(c) * addmm_params->fdc]);
        d = do_axpby ? addmm_params->alpha * d + addmm_params->beta * c_val
                     : d + c_val;
      }
    }
    Dout[i] = static_cast<T>(d);
  }

  tensor_t tD(D, extents_t(cols, rows), strides_t{1, params->ldd});
  if (inside) {
    auto d = tD.template slice<TSN, TSM>(0, 0);
    Dout.store(d);
  } else {
    Dout.store(tD);
  }
}
