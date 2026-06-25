#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define AT(TILE, IDX) TILE.thread_elements()[IDX]
#define SUB(TILE0, TILE1, TILE2)                \
  {                                             \
    AT(TILE0, 0) = AT(TILE1, 0) - AT(TILE2, 0); \
    AT(TILE0, 1) = AT(TILE1, 1) - AT(TILE2, 1); \
  }
#define FMA(TILE0, S, TILE1, TILE2)                 \
  {                                                 \
    AT(TILE0, 0) = S * AT(TILE1, 0) + AT(TILE2, 0); \
    AT(TILE0, 1) = S * AT(TILE1, 1) + AT(TILE2, 1); \
  }

#define SCALE(TILE0, S) \
  {                     \
    AT(TILE0, 0) *= S;  \
    AT(TILE0, 1) *= S;  \
  }
#define SCALE2(TILE0, S0, S1) \
  {                           \
    AT(TILE0, 0) *= S0;       \
    AT(TILE0, 1) *= S1;       \
  }
#define SCALE_TRI(TILE0, S0, S1)            \
  {                                         \
    AT(TILE0, 0) *= fn > fm ? 0.f : S0;     \
    AT(TILE0, 1) *= fn + 1 > fm ? 0.f : S1; \
  }
#define SCALE_TRIEQ(TILE0, S0, S1)           \
  {                                          \
    AT(TILE0, 0) *= fn >= fm ? 0.f : S0;     \
    AT(TILE0, 1) *= fn + 1 >= fm ? 0.f : S1; \
  }

template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_fused_chunk(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device StT* state_in [[buffer(3)]],
    const device InT* g [[buffer(4)]],
    const device InT* beta [[buffer(5)]],
    device InT* y [[buffer(6)]],
    device StT* state_out [[buffer(7)]],
    constant int& T [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  const short qid = thread_index_in_simdgroup / 4;
  const short fm = (qid & 4) +
      ((thread_index_in_simdgroup / 2) % 4); // row coordinate of the held tile
  const short fn = (qid & 2) * 2 +
      (thread_index_in_simdgroup % 2) * 2; // column coordinate of the held tile

  auto dv_idx = thread_position_in_grid.y * 8;

  // set up pointers
  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  y += b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  simdgroup_float8x8 S_tile[Dk / 8];

  // simdgroup matrices
  simdgroup_float8x8 V_tile, K_tile, KT_tile, Q_tile;
  simdgroup_float8x8 W_tile, U_tile;
  simdgroup_float8x8 WS_tile;
  simdgroup_float8x8 delta_tile;
  simdgroup_float8x8 tmp_tile;
  simdgroup_float8x8 QKt_tile;
  simdgroup_float8x8 out_tile;
  simdgroup_float8x8 KD_tile;

  // tiles for WY form computation
  simdgroup_float8x8 KKtK_tile, KKtV_tile, X_tile, KKt_tile;

  threadgroup float gamma[C];

  // load initial state into threadgroup
  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_load(S_tile[kk / 8], i_state + kk, Dk, ulong2(0, 0), true);
  }

  for (int t = 0; t < T; t += C) {
    float g_val = (thread_index_in_simdgroup < C)
        ? g_[thread_index_in_simdgroup * Hv + hv_idx]
        : 1.0f;

    float gamma_val = simd_prefix_inclusive_product(g_val);

    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = gamma_val;
    }

    float beta_fm = beta_[fm * Hv + hv_idx];

    KKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(K_tile, k_ + kk, Dk * Hk);
      simdgroup_load(KT_tile, k_ + kk, Dk * Hk, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(KKt_tile, K_tile, KT_tile, KKt_tile);
    }

    KKtK_tile = KKt_tile;
    KKtV_tile = KKt_tile;

    // elementwise multiplication by Gamma and beta (for V) and beta (for K)
    SCALE_TRIEQ(KKtK_tile, beta_fm, beta_fm)
    SCALE_TRIEQ(
        KKtV_tile,
        beta_fm * (gamma[fm] / gamma[fn]),
        beta_fm * (gamma[fm] / gamma[fn + 1]))

    WS_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    // Use the Neumann series: (I - T)^-1 = sum T^k, instead of doing forward
    // substitution to compute W (and U). For C=8 this is 8 matmuls, for generic
    // C this probably does not work well? Also fuse the WS in the same loop ->
    // less memory movements.
    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(K_tile, k_ + kk, Dk * Hk);
      SCALE(K_tile, beta_fm)
      W_tile = K_tile;
      for (int iter = 0; iter < C - 1; iter++) {
        simdgroup_multiply(X_tile, KKtK_tile, W_tile);
        SUB(W_tile, K_tile, X_tile)
      }

      SCALE(W_tile, gamma[fm])

      //  WS = W_left @ S^T
      simdgroup_multiply_accumulate(WS_tile, W_tile, S_tile[kk / 8], WS_tile);
    }

    simdgroup_load(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE(V_tile, beta_fm)
    U_tile = V_tile;
    for (int iter = 0; iter < C - 1; iter++) {
      simdgroup_multiply(X_tile, KKtV_tile, U_tile);
      SUB(U_tile, V_tile, X_tile)
    }

    // delta = U - WS
    SUB(delta_tile, U_tile, WS_tile)

    tmp_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    QKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(Q_tile, q_ + kk, Hk * Dk);
      simdgroup_load(K_tile, k_ + kk, Hk * Dk, ulong2(0, 0), true);

      SCALE(Q_tile, gamma[fm])

      // Q_left @ S^T
      simdgroup_multiply_accumulate(tmp_tile, Q_tile, S_tile[kk / 8], tmp_tile);
      simdgroup_multiply_accumulate(QKt_tile, Q_tile, K_tile, QKt_tile);
    }

    // (Q @ K) * Gamma, in the paper they use M here but it's probably a typo.
    SCALE_TRI(QKt_tile, (1.0f / gamma[fn]), (1.0f / gamma[fn + 1]))

    simdgroup_multiply_accumulate(out_tile, QKt_tile, delta_tile, tmp_tile);

    y[fm * Hv * Dv + dv_idx + fn] = static_cast<InT>(AT(out_tile, 0));
    y[fm * Hv * Dv + dv_idx + fn + 1] = static_cast<InT>(AT(out_tile, 1));

    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(K_tile, k_ + kk, Hk * Dk, ulong2(0, 0), true);

      SCALE2(K_tile, (gamma[C - 1] / gamma[fn]), (gamma[C - 1] / gamma[fn + 1]))

      simdgroup_multiply(KD_tile, K_tile, delta_tile);

      FMA(S_tile[kk / 8], gamma[C - 1], S_tile[kk / 8], KD_tile)
    }

    // advance pointers
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }

  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_store(S_tile[kk / 8], o_state + kk, Dk, ulong2(0, 0), true);
  }
}

/*
        auto grid   = MTL::Size(32, Dv, B * Hv);
    auto threads = MTL::Size(32, 4, 1);
 */
template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_seq(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device InT* g [[buffer(3)]], // [B, T, Hv] or [B, T, Hv, Dk]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device StT* state_in [[buffer(5)]], // [B, Hv, Dv, Dk]
    constant int& T [[buffer(6)]],
    device InT* y [[buffer(7)]], // [B, T, Hv, Dv]
    device StT* state_out [[buffer(8)]], // [B, Hv, Dv, Dk]
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  // kernel implementation
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);
  constexpr int n_per_t = Dk / 32;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  y += b_idx * T * Hv * Dv + hv_idx * Dv;

  auto dk_idx = thread_position_in_threadgroup.x;
  auto dv_idx = thread_position_in_grid.y;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(i_state[s_idx]);
  }

  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;
  auto beta_ = beta + b_idx * T * Hv;

  for (int t = 0; t < T; ++t) {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_[hv_idx];
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }
    // Increment data pointers to next time step
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    o_state[s_idx] = static_cast<StT>(state[i]);
  }
}
