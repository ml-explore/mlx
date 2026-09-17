// Copyright © 2024 Apple Inc.
#pragma once

#include <metal_atomic>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename InT, int Dk, int Dv, int Hk, int Hv, int Ckpt>
[[kernel]] void gated_delta_vjp_seq(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* b [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B*Hv, n_ckpt, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<float>* dq [[buffer(9)]],
    device mlx_atomic<float>* dk [[buffer(10)]],
    device float* dv [[buffer(11)]],
    device mlx_atomic<float>* dg [[buffer(12)]],
    device mlx_atomic<float>* db [[buffer(13)]],
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

  const int qk_stride = Hk * Dk;
  const int v_stride = Hv * Dv;
  const int g_stride = Hv;

  auto q_base = q + b_idx * T * qk_stride + hk_idx * Dk;
  auto k_base = k + b_idx * T * qk_stride + hk_idx * Dk;
  auto dq_base = dq + b_idx * T * qk_stride + hk_idx * Dk;
  auto dk_base = dk + b_idx * T * qk_stride + hk_idx * Dk;

  auto v_base = v + b_idx * T * v_stride + hv_idx * Dv;
  auto dv_base = dv + b_idx * T * v_stride + hv_idx * Dv;
  auto co_base = cot_o + b_idx * T * v_stride + hv_idx * Dv;

  auto g_base = g + b_idx * T * g_stride + hv_idx;
  auto b_base = b + b_idx * T * g_stride + hv_idx;
  auto dg_base = dg + b_idx * T * g_stride + hv_idx;
  auto db_base = db + b_idx * T * g_stride + hv_idx;

  const int n_ckpt = (T + Ckpt - 1) / Ckpt;

  float s_hat[n_per_t];
  auto base_state = cot_h + (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < n_per_t; i++) {
    s_hat[i] = base_state[n_per_t * dk_idx + i];
  }

  // Only this thread's n_per_t slice of the state is ever needed, so a whole
  // segment of entry states costs Ckpt * n_per_t registers and the replay needs
  // no device buffer.
  float seg[Ckpt][n_per_t];
  float s_prev[n_per_t];
  float s_dec[n_per_t];

  for (int seg_idx = n_ckpt - 1; seg_idx >= 0; --seg_idx) {
    const int t0 = seg_idx * Ckpt;
    const int seg_len = metal::min(Ckpt, T - t0);

    // Replay forward from the checkpoint, recording the entry state of each
    // step in the segment.
    auto c_state =
        state_cache + n * n_ckpt * Dv * Dk + seg_idx * Dv * Dk + dv_idx * Dk;

    float s[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      s[i] = c_state[n_per_t * dk_idx + i];
    }

    for (int j = 0; j < seg_len; ++j) {
      const int t = t0 + j;

      for (int i = 0; i < n_per_t; ++i) {
        seg[j][i] = s[i];
      }

      float gamma = static_cast<float>(g_base[t * g_stride]);
      float beta = static_cast<float>(b_base[t * g_stride]);

      auto k_t = k_base + t * qk_stride;

      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        const int s_idx = n_per_t * dk_idx + i;
        s[i] *= gamma;
        kv_mem += s[i] * static_cast<float>(k_t[s_idx]);
      }
      kv_mem = simd_sum(kv_mem);

      float delta =
          beta * (static_cast<float>(v_base[t * v_stride + dv_idx]) - kv_mem);

      for (int i = 0; i < n_per_t; ++i) {
        s[i] += delta * static_cast<float>(k_t[n_per_t * dk_idx + i]);
      }
    }

    // Backward over the same steps, newest first.
    for (int j = seg_len - 1; j >= 0; --j) {
      const int t = t0 + j;

      float gamma = static_cast<float>(g_base[t * g_stride]);
      float beta = static_cast<float>(b_base[t * g_stride]);

      auto q_t = q_base + t * qk_stride;
      auto k_t = k_base + t * qk_stride;
      auto dq_t = dq_base + t * qk_stride;
      auto dk_t = dk_base + t * qk_stride;

      float kv_mem = 0.0f;
      float co = static_cast<float>(co_base[t * v_stride + dv_idx]);
      float w = 0.0f;
      for (int i = 0; i < n_per_t; i++) {
        const int s_idx = n_per_t * dk_idx + i;
        s_prev[i] = seg[j][i];
        s_dec[i] = s_prev[i] * gamma;
        kv_mem += s_dec[i] * static_cast<float>(k_t[s_idx]);

        s_hat[i] += co * static_cast<float>(q_t[s_idx]);

        w += s_hat[i] * static_cast<float>(k_t[s_idx]);
      }
      kv_mem = simd_sum(kv_mem);
      w = simd_sum(w);

      if (thread_index_in_simdgroup == 0) {
        dv_base[t * v_stride + dv_idx] = beta * w;
      }

      float u = static_cast<float>(v_base[t * v_stride + dv_idx]) - kv_mem;
      float delta = beta * u;

      if (thread_index_in_simdgroup == 0) {
        mlx_atomic_fetch_add_explicit(db_base + t * g_stride, w * u, 0);
      }

      float dgamma = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;

        float s_t = s_dec[i] + delta * static_cast<float>(k_t[s_idx]);
        mlx_atomic_fetch_add_explicit(dq_t, co * s_t, s_idx);

        float contrib = beta * (u * s_hat[i] - w * s_dec[i]);
        mlx_atomic_fetch_add_explicit(dk_t, contrib, s_idx);

        s_hat[i] -= beta * w * static_cast<float>(k_t[s_idx]);

        dgamma += s_hat[i] * s_prev[i];
      }
      dgamma = simd_sum(dgamma);
      if (thread_index_in_simdgroup == 0) {
        mlx_atomic_fetch_add_explicit(dg_base + t * g_stride, dgamma, 0);
      }

      for (int i = 0; i < n_per_t; ++i) {
        s_hat[i] *= gamma;
      }
    }
  }

  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    dh[(n * Dv + dv_idx) * Dk + s_idx] = s_hat[i];
  }
}
