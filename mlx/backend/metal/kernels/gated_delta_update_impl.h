#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>

#define FULL_UNROLL _Pragma("clang loop unroll(full)")

#define AT(TILE, IDX) TILE.thread_elements()[IDX]
#define SUB(TILE0, TILE1, TILE2)                \
  {                                             \
    AT(TILE0, 0) = AT(TILE1, 0) - AT(TILE2, 0); \
    AT(TILE0, 1) = AT(TILE1, 1) - AT(TILE2, 1); \
  }
#define ADD(TILE0, TILE1, TILE2)                \
  {                                             \
    AT(TILE0, 0) = AT(TILE1, 0) + AT(TILE2, 0); \
    AT(TILE0, 1) = AT(TILE1, 1) + AT(TILE2, 1); \
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

// lambdas are not supported in metal 14 so porting to macros.

// non transposed
#define LOAD_M(M, SRC, LD, B)                                                  \
  if constexpr (B) {                                                           \
    AT(M, 0) =                                                                 \
        static_cast<float>((fm < valid_rows) ? ((SRC)[fm * (LD) + fn]) : 0.f); \
    AT(M, 1) = static_cast<float>(                                             \
        (fm < valid_rows) ? ((SRC)[fm * (LD) + fn + 1]) : 0.f);                \
  } else {                                                                     \
    AT(M, 0) = static_cast<float>((SRC)[fm * (LD) + fn]);                      \
    AT(M, 1) = static_cast<float>((SRC)[fm * (LD) + fn + 1]);                  \
  }

// transposed load: sequence is the column -> mask fn / fn+1
#define LOAD_MT(M, SRC, LD, B)                                                 \
  if constexpr (B) {                                                           \
    AT(M, 0) =                                                                 \
        static_cast<float>((fn < valid_rows) ? ((SRC)[fn * (LD) + fm]) : 0.f); \
    AT(M, 1) = static_cast<float>(                                             \
        (fn + 1 < valid_rows) ? ((SRC)[(fn + 1) * (LD) + fm]) : 0.f);          \
  } else {                                                                     \
    AT(M, 0) = static_cast<float>((SRC)[fn * (LD) + fm]);                      \
    AT(M, 1) = static_cast<float>((SRC)[(fn + 1) * (LD) + fm]);                \
  }

#define PROCESS_CHUNK_SG(B, S_tile, VALID)                                     \
  {                                                                            \
    const short valid_rows = (VALID);                                          \
                                                                               \
    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)               \
        ? metal::fast::log(                                                    \
              metal::max(                                                      \
                  static_cast<float>(                                          \
                      g_[thread_index_in_simdgroup * Hv + hv_idx]),            \
                  1e-6f))                                                      \
        : 0.0f;                                                                \
                                                                               \
    float gamma_val = simd_prefix_inclusive_sum(g_val);                        \
                                                                               \
    if (thread_index_in_simdgroup < C) {                                       \
      gamma[thread_index_in_simdgroup] = gamma_val;                            \
    }                                                                          \
    simdgroup_barrier(mem_flags::mem_threadgroup);                             \
                                                                               \
    float gamma_fm = metal::fast::exp(gamma[fm]);                              \
    float gamma_fmdfn = metal::fast::exp(gamma[fm] - gamma[fn]);               \
    float gamma_fmdfn1 = metal::fast::exp(gamma[fm] - gamma[fn + 1]);          \
    float gamma_Cdfn = metal::fast::exp(gamma[C - 1] - gamma[fn]);             \
    float gamma_Cdfn1 = metal::fast::exp(gamma[C - 1] - gamma[fn + 1]);        \
    float gamma_C = metal::fast::exp(gamma[C - 1]);                            \
                                                                               \
    float beta_fm = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;        \
                                                                               \
    KKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \
    FULL_UNROLL                                                                \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_M(K_tile, k_ + kk, Dk * Hk, B)                                      \
      LOAD_MT(KT_tile, k_ + kk, Dk * Hk, B)                                    \
      simdgroup_multiply_accumulate(KKt_tile, K_tile, KT_tile, KKt_tile);      \
    }                                                                          \
                                                                               \
    KKtK_tile = KKt_tile;                                                      \
    SCALE_TRIEQ(KKtK_tile, beta_fm, beta_fm)                                   \
                                                                               \
    simdgroup_float8x8 Tinv, P;                                                \
    AT(P, 0) = AT(KKtK_tile, 0);                                               \
    AT(P, 1) = AT(KKtK_tile, 1);                                               \
    SUB(Tinv, I_tile, KKtK_tile)                                               \
                                                                               \
    FULL_UNROLL                                                                \
    for (int step = 1; (1 << step) < C; step++) {                              \
      simdgroup_multiply(P, P, P);                                             \
      simdgroup_multiply_accumulate(Tinv, Tinv, P, Tinv);                      \
    }                                                                          \
                                                                               \
    WS_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                     \
    FULL_UNROLL                                                                \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_M(K_tile, k_ + kk, Dk * Hk, B)                                      \
      SCALE(K_tile, beta_fm)                                                   \
      simdgroup_multiply(W_tile, Tinv, K_tile);                                \
      SCALE(W_tile, gamma_fm)                                                  \
      simdgroup_multiply_accumulate(WS_tile, W_tile, S_tile[kk / 8], WS_tile); \
    }                                                                          \
                                                                               \
    SCALE_TRI(Tinv, gamma_fmdfn, gamma_fmdfn1)                                 \
                                                                               \
    LOAD_M(V_tile, v_ + dv_idx, Dv * Hv, B)                                    \
    SCALE(V_tile, beta_fm)                                                     \
    simdgroup_multiply(U_tile, Tinv, V_tile);                                  \
    SUB(delta_tile, U_tile, WS_tile)                                           \
                                                                               \
    tmp_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \
    QKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);                    \
    FULL_UNROLL                                                                \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_M(Q_tile, q_ + kk, Hk * Dk, B)                                      \
      LOAD_MT(K_tile, k_ + kk, Hk * Dk, B)                                     \
      simdgroup_multiply_accumulate(QKt_tile, Q_tile, K_tile, QKt_tile);       \
      SCALE(Q_tile, gamma_fm)                                                  \
      simdgroup_multiply_accumulate(                                           \
          tmp_tile, Q_tile, S_tile[kk / 8], tmp_tile);                         \
    }                                                                          \
                                                                               \
    SCALE_TRI(QKt_tile, gamma_fmdfn, gamma_fmdfn1)                             \
                                                                               \
    simdgroup_multiply_accumulate(out_tile, QKt_tile, delta_tile, tmp_tile);   \
                                                                               \
    if (fm < valid_rows) {                                                     \
      y[fm * Hv * Dv + dv_idx + fn] = static_cast<InT>(AT(out_tile, 0));       \
      y[fm * Hv * Dv + dv_idx + fn + 1] = static_cast<InT>(AT(out_tile, 1));   \
    }                                                                          \
                                                                               \
    FULL_UNROLL                                                                \
    for (int kk = 0; kk < Dk; kk += 8) {                                       \
      LOAD_MT(K_tile, k_ + kk, Hk * Dk, B)                                     \
      SCALE2(K_tile, gamma_Cdfn, gamma_Cdfn1)                                  \
      simdgroup_multiply(KD_tile, K_tile, delta_tile);                         \
      FMA(S_tile[kk / 8], gamma_C, S_tile[kk / 8], KD_tile)                    \
    }                                                                          \
  }

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_fused_chunk(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device float* state_in [[buffer(3)]],
    const device InT* g [[buffer(4)]],
    const device InT* beta [[buffer(5)]],
    device InT* y [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    constant int& T [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
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
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

#define OUTPUT(T)          \
  if (true) {              \
    simdgroup_store(T, y); \
    return;                \
  }
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
  simdgroup_float8x8 KKtK_tile, KKt_tile;

  threadgroup float gamma_all[C * 4];
  threadgroup float* gamma = gamma_all + sg_id * C;

  simdgroup_float8x8 I_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
  AT(I_tile, 0) = (fm == fn) ? 1.0f : 0.0f;
  AT(I_tile, 1) = (fm == fn + 1) ? 1.0f : 0.0f;

  // load initial state into registers
  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_load(S_tile[kk / 8], i_state + kk, Dk, ulong2(0, 0), true);
  }

  int t = 0;
  FULL_UNROLL
  for (; t + C <= T; t += C) {
    PROCESS_CHUNK_SG(false, S_tile, C);
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }
  if (t < T) {
    PROCESS_CHUNK_SG(true, S_tile, short(T - t));
  }

  FULL_UNROLL
  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_store(S_tile[kk / 8], o_state + kk, Dk, ulong2(0, 0), true);
  }
}

/*
        auto grid   = MTL::Size(32, Dv, B * Hv);
    auto threads = MTL::Size(32, 4, 1);
 */
template <typename InT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_seq(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device InT* g [[buffer(3)]], // [B, T, Hv] or [B, T, Hv, Dk]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device float* state_in [[buffer(5)]], // [B, Hv, Dv, Dk]
    constant int& T [[buffer(6)]],
    device InT* y [[buffer(7)]], // [B, T, Hv, Dv]
    device float* state_out [[buffer(8)]], // [B, Hv, Dv, Dk]
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
    o_state[s_idx] = static_cast<float>(state[i]);
  }
}