// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename InT, int Dv>
[[kernel]] void sdpa_vjp_odo(
    const device InT* o [[buffer(0)]], // [B, H, qL, Dv]
    const device InT* cot_o [[buffer(1)]], // [B, H, qL, Dv]
    device float* odo [[buffer(2)]], // [B, H, qL]
    constant int& qL [[buffer(3)]],
    uint3 tpg [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  const uint bh = tpg.z;
  const int i = int(tpg.y);

  if (i >= qL) {
    return;
  }

  const size_t row = (size_t)bh * qL + i;
  auto o_i = o + row * Dv;
  auto co_i = cot_o + row * Dv;

  float acc = 0.0f;
  for (int d = int(simd_lane_id); d < Dv; d += 32) {
    acc += float(o_i[d]) * float(co_i[d]);
  }
  acc = simd_sum(acc);

  if (simd_lane_id == 0) {
    odo[row] = acc;
  }
}

struct SDPAVJPTileParams {
  int bq;
  int bk;
  int qL;
  int i0;
  int j0;
  float scale;
  int diag_off;
  int causal;
};

template <typename T>
[[kernel]] void sdpa_vjp_ds(
    const device T* S [[buffer(0)]], // [BH, bq, bk]
    const device T* dP [[buffer(1)]], // [BH, bq, bk]
    const device float* lse [[buffer(2)]], // [B, H, qL]
    const device float* odo [[buffer(3)]], // [B, H, qL]
    device T* dS [[buffer(4)]], // [BH, bq, bk]
    device T* dSt [[buffer(5)]], // [BH, bk, bq]
    device T* Pt [[buffer(6)]], // [BH, bk, bq]
    const constant SDPAVJPTileParams& p [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
  int col = int(gid.x);
  int row = int(gid.y);
  int bh = int(gid.z);
  if (row >= p.bq || col >= p.bk) {
    return;
  }

  size_t sbase = (size_t(bh) * size_t(p.bq) + size_t(row)) * size_t(p.bk);
  size_t tbase = (size_t(bh) * size_t(p.bk) + size_t(col)) * size_t(p.bq);
  size_t qi = size_t(bh) * size_t(p.qL) + size_t(p.i0 + row);

  float l = lse[qi];
  float dlt = odo[qi];
  // Fully masked rows carry lse == -inf; exp(-inf - -inf) would be NaN.
  bool dead = (l < 0.0f) && metal::isinf(l);
  int lim = p.i0 + row + p.diag_off;

  float pv;
  if (dead || (p.causal != 0 && (p.j0 + col) > lim)) {
    pv = 0.0f;
  } else {
    pv = metal::fast::exp(static_cast<float>(S[sbase + col]) * p.scale - l);
  }
  float dsv = pv * (static_cast<float>(dP[sbase + col]) - dlt) * p.scale;

  dS[sbase + col] = static_cast<T>(dsv);
  dSt[tbase + row] = static_cast<T>(dsv);
  Pt[tbase + row] = static_cast<T>(pv);
}

template <typename T, bool Accum>
[[kernel]] void sdpa_vjp_reduce(
    const device T* src [[buffer(0)]],
    device float* acc [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant int& rows [[buffer(3)]],
    const constant int& dim [[buffer(4)]],
    const constant int& group [[buffer(5)]],
    const constant int& acc_rows [[buffer(6)]],
    const constant int& row_off [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
  int c = int(gid.x);
  int row = int(gid.y);
  int bh = int(gid.z);
  if (c >= dim || row >= rows) {
    return;
  }

  size_t sbase = size_t(bh) * size_t(group) * size_t(rows) * size_t(dim);
  float sum = 0.0f;
  for (int g = 0; g < group; ++g) {
    size_t si = sbase +
        (size_t(g) * size_t(rows) + size_t(row)) * size_t(dim) + size_t(c);
    sum += static_cast<float>(src[si]);
  }

  size_t ai =
      (size_t(bh) * size_t(acc_rows) + size_t(row_off + row)) * size_t(dim) +
      size_t(c);
  float total = Accum ? acc[ai] + sum : sum;
  acc[ai] = total;
  out[ai] = static_cast<T>(total);
}

// Instantiations
#define instantiate_odo(in_type, dv) \
  instantiate_kernel(                \
      "sdpa_vjp_odo_" #in_type "_" #dv, sdpa_vjp_odo, in_type, dv)

#define instantiate_odo_shapes(in_type) \
  instantiate_odo(in_type, 64);         \
  instantiate_odo(in_type, 72);         \
  instantiate_odo(in_type, 80);         \
  instantiate_odo(in_type, 96);         \
  instantiate_odo(in_type, 128);        \
  instantiate_odo(in_type, 192);        \
  instantiate_odo(in_type, 256);

instantiate_odo_shapes(bfloat16_t);
instantiate_odo_shapes(float16_t);
instantiate_odo_shapes(float);

#define instantiate_sdpa_vjp_ds(tname, type) \
  instantiate_kernel("sdpa_vjp_ds_" #tname, sdpa_vjp_ds, type)

instantiate_sdpa_vjp_ds(float, float);
instantiate_sdpa_vjp_ds(float16_t, float16_t);
instantiate_sdpa_vjp_ds(bfloat16_t, bfloat16_t);

#define instantiate_sdpa_vjp_reduce(tname, type)                 \
  instantiate_kernel(                                            \
      "sdpa_vjp_reduce_add_" #tname, sdpa_vjp_reduce, type, true) \
  instantiate_kernel(                                            \
      "sdpa_vjp_reduce_set_" #tname, sdpa_vjp_reduce, type, false)

instantiate_sdpa_vjp_reduce(float, float);
instantiate_sdpa_vjp_reduce(float16_t, float16_t);
instantiate_sdpa_vjp_reduce(bfloat16_t, bfloat16_t);
