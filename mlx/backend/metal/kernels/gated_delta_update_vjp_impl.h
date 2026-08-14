#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

#include <metal_atomic>
#include "mlx/backend/metal/kernels/atomic.h"

#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename InT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_vjp_seq(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* b [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B*Hv, T, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<InT>* dq [[buffer(9)]], // [B, T, Hk, Dk]
    device mlx_atomic<InT>* dk [[buffer(10)]],
    device InT* dv [[buffer(11)]],
    device mlx_atomic<InT>* dg [[buffer(12)]],
    device mlx_atomic<InT>* db [[buffer(13)]],
    device float* dh [[buffer(14)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);
  constexpr int n_per_t = Dk / 32;

  auto dk_idx = thread_position_in_threadgroup.x;
  auto dv_idx = thread_position_in_grid.y;

  // Starting from the last timestep (T-1)
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk + (T - 1) * Hk * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk + (T - 1) * Hk * Dk;
  auto dq_ = dq + b_idx * T * Hk * Dk + hk_idx * Dk + (T - 1) * Hk * Dk;
  auto dk_ = dk + b_idx * T * Hk * Dk + hk_idx * Dk + (T - 1) * Hk * Dk;

  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv + (T - 1) * Hv * Dv;
  auto dv_ = dv + b_idx * T * Hv * Dv + hv_idx * Dv + (T - 1) * Hv * Dv;

  auto co_ = cot_o + b_idx * T * Hv * Dv + hv_idx * Dv + (T - 1) * Hv * Dv;

  auto g_ = g + b_idx * T * Hv + (T - 1) * Hv + hv_idx;
  auto b_ = b + b_idx * T * Hv + (T - 1) * Hv + hv_idx;
  auto dg_ = dg + b_idx * T * Hv + (T - 1) * Hv + hv_idx;
  auto db_ = db + b_idx * T * Hv + (T - 1) * Hv + hv_idx;

  auto c_state =
      state_cache + n * T * Dv * Dk + (T - 1) * Dv * Dk + dv_idx * Dk;

  float s_hat[n_per_t]; // gradient
  float s_prev[n_per_t]; // state at entry
  float s_dec[n_per_t]; // state * gamma

  auto base_state = cot_h + (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < n_per_t; i++) {
    s_hat[i] = base_state[n_per_t * dk_idx + i];
  }

  for (int t = T - 1; t >= 0; --t) {
    float gamma = static_cast<float>(*g_);
    float beta = static_cast<float>(*b_);

    // Recompute forward state: s_prev = cache[t], s_dec = s_prev * gamma
    float kv_mem = 0.0f;
    float co = static_cast<float>(co_[dv_idx]);
    float w = 0.0f;
    for (int i = 0; i < n_per_t; i++) {
      const int s_idx = n_per_t * dk_idx + i;
      s_prev[i] = c_state[s_idx];
      s_dec[i] = s_prev[i] * gamma;
      kv_mem += s_dec[i] * static_cast<float>(k_[s_idx]);

      // s_hat += outer(co, q)
      s_hat[i] += co * static_cast<float>(q_[s_idx]);

      // w = dot(s_hat, k)
      w += s_hat[i] * static_cast<float>(k_[s_idx]);
    }
    kv_mem = simd_sum(kv_mem);
    w = simd_sum(w);

    if (thread_index_in_simdgroup == 0) {
      dv_[dv_idx] = static_cast<InT>(beta * w);
    }

    float u = static_cast<float>(v_[dv_idx]) - kv_mem;
    float delta = beta * u;

    // dv = beta * w
    if (thread_index_in_simdgroup == 0) {
      dv_[dv_idx] = static_cast<InT>(beta * w);
    }

    // dbeta = dot(w, u)
    if (thread_index_in_simdgroup == 0) {
      mlx_atomic_fetch_add_explicit(db_, static_cast<InT>(w * u), 0);
      // db_[0] = static_cast<InT>(w * u);
    }

    float dgamma = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;

      // dq = S_t^T co,  S_t = s_dec + delta * k
      float s_t = s_dec[i] + delta * static_cast<float>(k_[s_idx]);
      mlx_atomic_fetch_add_explicit(dq_, static_cast<InT>(co * s_t), s_idx);

      // dk += beta * (u * s_hat - w * s_dec)
      float contrib = beta * (u * s_hat[i] - w * s_dec[i]);
      mlx_atomic_fetch_add_explicit(dk_, static_cast<InT>(contrib), s_idx);

      // s_hat -= beta * w * k
      s_hat[i] -= beta * w * static_cast<float>(k_[s_idx]);

      // dg = dot(s_hat, s_prev)
      dgamma += s_hat[i] * s_prev[i];
    }
    dgamma = simd_sum(dgamma);
    if (thread_index_in_simdgroup == 0) {
      mlx_atomic_fetch_add_explicit(dg_, static_cast<InT>(dgamma), 0);
    }

    // s_hat *= gamma
    for (int i = 0; i < n_per_t; ++i) {
      s_hat[i] *= gamma;
    }

    // Decrement to previous timestep
    q_ -= Hk * Dk;
    k_ -= Hk * Dk;
    v_ -= Hv * Dv;
    co_ -= Hv * Dv;
    g_ -= Hv;
    b_ -= Hv;
    dq_ -= Hk * Dk;
    dk_ -= Hk * Dk;
    dv_ -= Hv * Dv;
    dg_ -= Hv;
    db_ -= Hv;
    c_state -= Dv * Dk;
  }
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    dh[(n * Dv + dv_idx) * Dk + s_idx] = s_hat[i];
  }
}