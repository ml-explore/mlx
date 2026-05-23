// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/kq_quantized.h"

template <typename T>
METAL_FUNC void kq_q8_0_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    uint gid) {
  if (gid >= num_blocks) {
    return;
  }
  const device T* x = w + gid * KQ_Q8_0_GROUP;
  device uint8_t* block_addr = out + gid * KQ_Q8_0_BLOCK_BYTES;

  float amax = 0.0f;
  for (int j = 0; j < KQ_Q8_0_GROUP; j++) {
    amax = max(amax, fabs(float(x[j])));
  }

  const float d = amax / 127.0f;
  const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

  *(device half*)(block_addr + KQ_Q8_0_D_OFFSET) = half(d);

  device int8_t* qs = (device int8_t*)(block_addr + KQ_Q8_0_Q_OFFSET);
  for (int j = 0; j < KQ_Q8_0_GROUP; j++) {
    float v = float(x[j]) * id;
    qs[j] = int8_t(clamp(round(v), -127.0f, 127.0f));
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q8_0_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_Q8_0_GROUP, "Q8_0 requires group_size=32");
  static_assert(bits == 8, "Q8_0 requires bits=8");
  (void)imatrix;
  (void)has_imatrix;
  kq_q8_0_quantize_impl<T>(w, out, num_blocks, gid);
}

inline int kq_nearest_int(float v) {
  return int(rint(v));
}

template <int N>
inline float kq_make_qp_quants(
    const threadgroup float* x,
    const threadgroup float* qw,
    int nmax,
    thread uint8_t* L_out) {
  float max_v = 0.0f;
  for (int i = 0; i < N; i++) {
    if (x[i] > max_v)
      max_v = x[i];
  }
  if (max_v < 1e-30f) { // GROUP_MAX_EPS
    for (int i = 0; i < N; i++)
      L_out[i] = 0;
    return 0.0f;
  }
  float iscale = float(nmax) / max_v;
  uint8_t L[N];
  for (int i = 0; i < N; i++) {
    int l = kq_nearest_int(iscale * x[i]);
    L[i] = uint8_t(max(0, min(nmax, l)));
  }
  float scale = 1.0f / iscale;
  float best_mse = 0.0f;
  for (int i = 0; i < N; i++) {
    float diff = x[i] - scale * float(L[i]);
    best_mse += qw[i] * diff * diff;
  }
  for (int is = -4; is <= 4; is++) {
    if (is == 0)
      continue;
    float iscale_is = (0.1f * float(is) + float(nmax)) / max_v;
    float scale_is = 1.0f / iscale_is;
    float mse = 0.0f;
    for (int i = 0; i < N; i++) {
      int l = kq_nearest_int(iscale_is * x[i]);
      l = min(nmax, l);
      float diff = x[i] - scale_is * float(l);
      mse += qw[i] * diff * diff;
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0.0f, suml2 = 0.0f;
  for (int i = 0; i < N; i++) {
    int l = kq_nearest_int(iscale * x[i]);
    l = min(nmax, l);
    L[i] = uint8_t(l);
    sumlx += qw[i] * x[i] * float(l);
    suml2 += qw[i] * float(l) * float(l);
  }
  for (int itry = 0; itry < 5; itry++) {
    int n_changed = 0;
    for (int i = 0; i < N; i++) {
      float w = qw[i];
      float slx = sumlx - w * x[i] * float(L[i]);
      float sl2 = suml2 - w * float(L[i]) * float(L[i]);
      if (slx > 0.0f && sl2 > 0.0f) {
        int new_l = kq_nearest_int(x[i] * sl2 / slx);
        new_l = min(nmax, new_l);
        if (new_l != int(L[i])) {
          slx += w * x[i] * float(new_l);
          sl2 += w * float(new_l) * float(new_l);
          if (slx * slx * suml2 > sumlx * sumlx * sl2) {
            L[i] = uint8_t(new_l);
            sumlx = slx;
            suml2 = sl2;
            n_changed++;
          }
        }
      }
    }
    if (n_changed == 0)
      break;
  }
  for (int i = 0; i < N; i++)
    L_out[i] = L[i];
  return (suml2 > 0.0f) ? (sumlx / suml2) : 0.0f;
}

template <int N>
inline float kq_make_qkx3_quants(
    const threadgroup float* x,
    const threadgroup float* qw,
    int nmax,
    thread float& the_min) {
  const float rmin = -0.9f;
  const float rdelta = 0.05f;
  const int nstep = 36;

  float min_v = x[0];
  float max_v = x[0];
  float sum_w = qw[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < N; i++) {
    if (x[i] < min_v)
      min_v = x[i];
    if (x[i] > max_v)
      max_v = x[i];
    float w = qw[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min_v > 0.0f)
    min_v = 0.0f;
  if (max_v <= min_v) {
    the_min = -min_v;
    return 0.0f;
  }

  float iscale = float(nmax) / (max_v - min_v);
  float scale = 1.0f / iscale;
  uint8_t L[N];
  float best_mad = 0.0f;
  for (int i = 0; i < N; i++) {
    int l = kq_nearest_int(iscale * (x[i] - min_v));
    L[i] = uint8_t(max(0, min(nmax, l)));
    float diff = scale * float(L[i]) + min_v - x[i];
    best_mad += qw[i] * diff * diff;
  }
  float best_min = min_v;

  uint8_t Laux[N];
  for (int is = 0; is <= nstep; is++) {
    float iscale_is =
        (rmin + rdelta * float(is) + float(nmax)) / (max_v - min_v);
    float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
    for (int i = 0; i < N; i++) {
      int l = kq_nearest_int(iscale_is * (x[i] - min_v));
      l = max(0, min(nmax, l));
      Laux[i] = uint8_t(l);
      float w = qw[i];
      sum_l += w * float(l);
      sum_l2 += w * float(l) * float(l);
      sum_xl += w * float(l) * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0.0f) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0.0f) {
        this_min = 0.0f;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0.0f;
      for (int i = 0; i < N; i++) {
        float diff = this_scale * float(Laux[i]) + this_min - x[i];
        mad += qw[i] * diff * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < N; i++)
          L[i] = Laux[i];
        best_mad = mad;
        scale = this_scale;
        best_min = this_min;
      }
    }
  }
  the_min = -best_min;
  return scale;
}

inline void kq_compute_sigma2_av_x(
    threadgroup const float* Xs,
    threadgroup float* scratch,
    uint lid,
    uint simd_id,
    uint lane_id,
    float factor = 2.0f) {
  float my_x = Xs[lid];
  float simd_x2 = simd_sum(my_x * my_x);
  if (lane_id == 0)
    scratch[simd_id] = simd_x2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    float v = (lane_id < 8) ? scratch[lane_id] : 0.0f;
    float total = simd_sum(v);
    if (lane_id == 0) {
      float sigma2 = factor * total / 256.0f;
      scratch[8] = sigma2;
      scratch[9] = sqrt(sigma2);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void kq_pack_scales12(
    device uint8_t* scales12,
    const thread uint8_t* Ls,
    const thread uint8_t* Lm) {
  for (int i = 0; i < 12; i++)
    scales12[i] = 0;
  for (int j = 0; j < 8; j++) {
    uint8_t ls = Ls[j];
    uint8_t lm = Lm[j];
    if (j < 4) {
      scales12[j] = ls;
      scales12[j + 4] = lm;
    } else {
      scales12[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
      scales12[j - 4] |= ((ls >> 4) << 6);
      scales12[j] |= ((lm >> 4) << 6);
    }
  }
}

inline void kq_unpack_scale_min_k4(
    int j,
    const device uint8_t* scales12,
    thread uint8_t& sc_out,
    thread uint8_t& mn_out) {
  if (j < 4) {
    sc_out = scales12[j] & 0x3F;
    mn_out = scales12[j + 4] & 0x3F;
  } else {
    sc_out = (scales12[j + 4] & 0x0F) | ((scales12[j - 4] >> 6) << 4);
    mn_out = (scales12[j + 4] >> 4) | ((scales12[j] >> 6) << 4);
  }
}

template <typename T, int bits>
METAL_FUNC void kq_q45_k_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    const device float* imatrix,
    bool has_imatrix,
    uint K,
    uint tg_id,
    uint lid,
    uint simd_id,
    uint lane_id,
    threadgroup float* Xs,
    threadgroup float* QWs,
    threadgroup uint8_t* L_tgm,
    threadgroup float* scales_sb,
    threadgroup float* mins_sb,
    threadgroup float* sw_sb,
    threadgroup uint8_t* Ls,
    threadgroup uint8_t* Lm,
    threadgroup float* scratch) {
  static_assert(bits == 4 || bits == 5, "shared Q4_K/Q5_K only");
  constexpr int nmax = (bits == 4) ? 15 : 31;
  constexpr int block_bytes =
      (bits == 4) ? KQ_Q4_K_BLOCK_BYTES : KQ_Q5_K_BLOCK_BYTES;
  constexpr int d_off = (bits == 4) ? KQ_Q4_K_D_OFFSET : KQ_Q5_K_D_OFFSET;
  constexpr int dmin_off =
      (bits == 4) ? KQ_Q4_K_DMIN_OFFSET : KQ_Q5_K_DMIN_OFFSET;
  constexpr int sc_off =
      (bits == 4) ? KQ_Q4_K_SCALES_OFFSET : KQ_Q5_K_SCALES_OFFSET;
  constexpr int qs_off = (bits == 4) ? KQ_Q4_K_QS_OFFSET : KQ_Q5_K_QS_OFFSET;
  constexpr int superblock = 256;

  if (tg_id >= num_blocks)
    return;

  device uint8_t* block_addr = out + tg_id * block_bytes;
  const device T* x_global = w + tg_id * superblock;

  // -- Phase 1: Load Xs[256] --
  Xs[lid] = float(x_global[lid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  kq_compute_sigma2_av_x(Xs, scratch, lid, simd_id, lane_id);
  float sigma2 = scratch[8];
  float av_x = scratch[9];

  // -- Phase 2: weights --
  if (has_imatrix) {
    uint k_off = (tg_id * superblock) % K;
    QWs[lid] = imatrix[k_off + lid] * sqrt(sigma2 + Xs[lid] * Xs[lid]);
  } else {
    QWs[lid] = av_x + abs(Xs[lid]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 3: per-sub-block fit --
  if (lane_id == 0) {
    int sb_off = simd_id * 32;
    float sumw = 0.0f;
    for (int l = 0; l < 32; l++)
      sumw += QWs[sb_off + l];
    sw_sb[simd_id] = sumw;
    float the_min;
    float scale =
        kq_make_qkx3_quants<32>(&Xs[sb_off], &QWs[sb_off], nmax, the_min);
    scales_sb[simd_id] = scale;
    mins_sb[simd_id] = the_min;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 4: super-scale fit + pack scales[12] --
  if (simd_id == 0 && lane_id == 0) {
    uint8_t Ls_local[8];
    uint8_t Lm_local[8];
    float d_block =
        kq_make_qp_quants<8>(&scales_sb[0], &sw_sb[0], 63, Ls_local);
    float m_block = kq_make_qp_quants<8>(&mins_sb[0], &sw_sb[0], 63, Lm_local);
    for (int i = 0; i < 8; i++) {
      Ls[i] = Ls_local[i];
      Lm[i] = Lm_local[i];
    }
    *(device half*)(block_addr + d_off) = half(d_block);
    *(device half*)(block_addr + dmin_off) = half(m_block);
    kq_pack_scales12(block_addr + sc_off, Ls_local, Lm_local);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 5: re-quantize --
  float d_wire = float(*(device half*)(block_addr + d_off));
  float dmin_wire = float(*(device half*)(block_addr + dmin_off));
  int my_sb = int(lid) / 32;
  uint8_t sc, mn;
  kq_unpack_scale_min_k4(my_sb, block_addr + sc_off, sc, mn);
  float d_final = d_wire * float(sc);
  float dm_final = dmin_wire * float(mn);
  uint8_t my_L;
  if (d_final == 0.0f) {
    my_L = 0;
  } else {
    int l = kq_nearest_int((Xs[lid] + dm_final) / d_final);
    l = max(0, min(nmax, l));
    my_L = uint8_t(l);
  }
  L_tgm[lid] = my_L;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 6: pack qs + qh --
  if (lid < 128) {
    int stride = int(lid) / 32;
    int l = int(lid) % 32;
    uint8_t lo = L_tgm[64 * stride + l] & 0x0F;
    uint8_t hi = L_tgm[64 * stride + l + 32] & 0x0F;
    device uint8_t* qs = block_addr + qs_off;
    qs[32 * stride + l] = lo | (hi << 4);
  }
  if (bits == 5 && lid >= 128 && lid < 160) {
    int j = int(lid) - 128;
    uint8_t b = 0;
    for (int block_idx = 0; block_idx < 8; block_idx++) {
      if (L_tgm[block_idx * 32 + j] > 15) {
        b |= uint8_t(1 << block_idx);
      }
    }
    device uint8_t* qh = block_addr + KQ_Q5_K_QH_OFFSET;
    qh[j] = b;
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q4_k_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    const constant uint& K [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]) {
  static_assert(group_size == 256, "Q4_K requires group_size=256");
  static_assert(bits == 4, "Q4_K requires bits=4");
  threadgroup float Xs[256];
  threadgroup float QWs[256];
  threadgroup uint8_t L_tgm[256];
  threadgroup float scales_sb[8];
  threadgroup float mins_sb[8];
  threadgroup float sw_sb[8];
  threadgroup uint8_t Ls[8];
  threadgroup uint8_t Lm[8];
  threadgroup float scratch[16];
  kq_q45_k_quantize_impl<T, 4>(
      w,
      out,
      num_blocks,
      imatrix,
      has_imatrix != 0,
      K,
      tg_id,
      lid,
      simd_id,
      lane_id,
      Xs,
      QWs,
      L_tgm,
      scales_sb,
      mins_sb,
      sw_sb,
      Ls,
      Lm,
      scratch);
}

inline float kq_make_qx_quants_16(
    const threadgroup float* x,
    const threadgroup float* qw,
    int nmax,
    thread uint8_t* L_out) {
  const int n = 16;
  float max_v = 0.0f;
  float amax = 0.0f;
  for (int i = 0; i < n; i++) {
    float ax = abs(x[i]);
    if (ax > amax) {
      amax = ax;
      max_v = x[i];
    }
  }
  if (amax < 1e-30f) {
    for (int i = 0; i < n; i++)
      L_out[i] = uint8_t(nmax);
    return 0.0f;
  }
  float iscale = -float(nmax) / max_v;
  float sumlx = 0.0f, suml2 = 0.0f;
  for (int i = 0; i < n; i++) {
    int l = kq_nearest_int(iscale * x[i]);
    l = max(-nmax, min(nmax - 1, l));
    L_out[i] = uint8_t(l + nmax);
    float w = (qw != nullptr) ? qw[i] : x[i] * x[i];
    sumlx += w * x[i] * float(l);
    suml2 += w * float(l) * float(l);
  }
  float scale = (suml2 > 0.0f) ? (sumlx / suml2) : 0.0f;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; is++) {
    if (is == 0)
      continue;
    float iscale_is = -(float(nmax) + 0.1f * float(is)) / max_v;
    float slx = 0.0f, sl2 = 0.0f;
    for (int i = 0; i < n; i++) {
      int l = kq_nearest_int(iscale_is * x[i]);
      l = max(-nmax, min(nmax - 1, l));
      float w = (qw != nullptr) ? qw[i] : x[i] * x[i];
      slx += w * x[i] * float(l);
      sl2 += w * float(l) * float(l);
    }
    if (sl2 > 0.0f && slx * slx > best * sl2) {
      for (int i = 0; i < n; i++) {
        int l = kq_nearest_int(iscale_is * x[i]);
        l = max(-nmax, min(nmax - 1, l));
        L_out[i] = uint8_t(l + nmax);
      }
      scale = slx / sl2;
      best = scale * slx;
    }
  }
  return scale;
}

template <typename T>
METAL_FUNC void kq_q6_k_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    const device float* imatrix,
    bool has_imatrix,
    uint K,
    uint tg_id,
    uint lid,
    uint simd_id,
    uint lane_id,
    threadgroup float* Xs,
    threadgroup float* QWs,
    threadgroup uint8_t* L_tgm,
    threadgroup float* scales_sb,
    threadgroup float* scratch) {
  if (tg_id >= num_blocks) {
    return;
  }

  device uint8_t* block_addr = out + tg_id * KQ_Q6_K_BLOCK_BYTES;
  const device T* x_global = w + tg_id * KQ_Q6_K_SUPERBLOCK;

  // -- Phase 1: Load Xs[256] --
  Xs[lid] = float(x_global[lid]);
  if (has_imatrix) {
    uint k_off = (tg_id * KQ_Q6_K_SUPERBLOCK) % K;
    QWs[lid] = imatrix[k_off + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 2: per-sub-block fit --
  if ((lid % 16) == 0) {
    int sb = int(lid) / 16;
    int sb_off = sb * 16;
    uint8_t L_local[16];
    const threadgroup float* qw_ptr =
        has_imatrix ? &QWs[sb_off] : (const threadgroup float*)nullptr;
    float scale = kq_make_qx_quants_16(&Xs[sb_off], qw_ptr, 32, L_local);
    scales_sb[sb] = scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 3: super-scale + write d, scales[16] --
  if (simd_id == 0 && lane_id == 0) {
    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
      float s = scales_sb[sb];
      float a = abs(s);
      if (a > max_abs_scale) {
        max_abs_scale = a;
        max_scale = s;
      }
    }
    if (max_abs_scale < 1e-30f) {
      scratch[0] = -1.0f;
    } else {
      float iscale = -128.0f / max_scale;
      *(device half*)(block_addr + KQ_Q6_K_D_OFFSET) = half(1.0f / iscale);
      device int8_t* scales_out =
          (device int8_t*)(block_addr + KQ_Q6_K_SCALES_OFFSET);
      for (int sb = 0; sb < 16; sb++) {
        int s = kq_nearest_int(iscale * scales_sb[sb]);
        s = min(127, s);
        scales_out[sb] = int8_t(s);
      }
      scratch[0] = 1.0f;
      scratch[1] = float(*(device half*)(block_addr + KQ_Q6_K_D_OFFSET));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bool all_zero = scratch[0] < 0.0f;
  if (all_zero) {
    if (lid < uint(KQ_Q6_K_BLOCK_BYTES)) {
      block_addr[lid] = 0;
    }
    return;
  }

  float d_wire = scratch[1];

  // -- Phase 4: re-quantize --
  int my_sb = int(lid) / 16;
  const device int8_t* scales_out =
      (const device int8_t*)(block_addr + KQ_Q6_K_SCALES_OFFSET);
  int8_t s_int8 = scales_out[my_sb];
  float d_eff = d_wire * float(s_int8);

  uint8_t my_L;
  if (d_eff == 0.0f) {
    my_L = 32;
  } else {
    float xv = Xs[lid];
    int l = kq_nearest_int(xv / d_eff);
    l = max(-32, min(31, l));
    my_L = uint8_t(l + 32);
  }
  L_tgm[lid] = my_L;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 5: pack ql[128] + qh[64] --
  if (lid < 64) {
    int stride = int(lid) / 32;
    int l = int(lid) % 32;
    int base = stride * 128;
    uint8_t La = L_tgm[base + l];
    uint8_t Lb = L_tgm[base + l + 32];
    uint8_t Lc = L_tgm[base + l + 64];
    uint8_t Ld = L_tgm[base + l + 96];
    device uint8_t* ql_out = block_addr + KQ_Q6_K_QL_OFFSET + stride * 64;
    device uint8_t* qh_out = block_addr + KQ_Q6_K_QH_OFFSET + stride * 32;
    ql_out[l] = (La & 0x0F) | ((Lc & 0x0F) << 4);
    ql_out[l + 32] = (Lb & 0x0F) | ((Ld & 0x0F) << 4);
    qh_out[l] =
        (La >> 4) | ((Lb >> 4) << 2) | ((Lc >> 4) << 4) | ((Ld >> 4) << 6);
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q6_k_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    const constant uint& K [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]) {
  static_assert(group_size == 256, "Q6_K requires group_size=256");
  static_assert(bits == 6, "Q6_K requires bits=6");
  threadgroup float Xs[256];
  threadgroup float QWs[256];
  threadgroup uint8_t L_tgm[256];
  threadgroup float scales_sb[16];
  threadgroup float scratch[16];
  kq_q6_k_quantize_impl<T>(
      w,
      out,
      num_blocks,
      imatrix,
      has_imatrix != 0,
      K,
      tg_id,
      lid,
      simd_id,
      lane_id,
      Xs,
      QWs,
      L_tgm,
      scales_sb,
      scratch);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_k_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    const constant uint& K [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]) {
  static_assert(group_size == 256, "Q5_K requires group_size=256");
  static_assert(bits == 5, "Q5_K requires bits=5");
  threadgroup float Xs[256];
  threadgroup float QWs[256];
  threadgroup uint8_t L_tgm[256];
  threadgroup float scales_sb[8];
  threadgroup float mins_sb[8];
  threadgroup float sw_sb[8];
  threadgroup uint8_t Ls[8];
  threadgroup uint8_t Lm[8];
  threadgroup float scratch[16];
  kq_q45_k_quantize_impl<T, 5>(
      w,
      out,
      num_blocks,
      imatrix,
      has_imatrix != 0,
      K,
      tg_id,
      lid,
      simd_id,
      lane_id,
      Xs,
      QWs,
      L_tgm,
      scales_sb,
      mins_sb,
      sw_sb,
      Ls,
      Lm,
      scratch);
}

template <typename T>
METAL_FUNC void kq_q3_k_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    const device float* imatrix,
    bool has_imatrix,
    uint K,
    uint tg_id,
    uint lid,
    uint simd_id,
    uint lane_id,
    threadgroup float* Xs,
    threadgroup float* QWs,
    threadgroup uint8_t* L_tgm,
    threadgroup float* scales_sb,
    threadgroup float* scratch) {
  if (tg_id >= num_blocks) {
    return;
  }

  device uint8_t* block_addr = out + tg_id * KQ_Q3_K_BLOCK_BYTES;
  const device T* x_global = w + tg_id * KQ_Q3_K_SUPERBLOCK;

  // -- Phase 1: Load Xs[256] --
  Xs[lid] = float(x_global[lid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 2: imatrix weights --
  if (has_imatrix) {
    kq_compute_sigma2_av_x(Xs, scratch, lid, simd_id, lane_id);
    float sigma2 = scratch[8];
    uint k_off = (tg_id * KQ_Q3_K_SUPERBLOCK) % K;
    QWs[lid] = imatrix[k_off + lid] * sqrt(sigma2 + Xs[lid] * Xs[lid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // -- Phase 3: per-sub-block fit --
  if ((lid % 16) == 0) {
    int sb = int(lid) / 16;
    int sb_off = sb * 16;
    uint8_t L_local[16];
    const threadgroup float* qw_ptr =
        has_imatrix ? &QWs[sb_off] : (const threadgroup float*)nullptr;
    float scale = kq_make_qx_quants_16(&Xs[sb_off], qw_ptr, 4, L_local);
    scales_sb[sb] = scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 4: super-scale + pack scales[12] --
  if (simd_id == 0 && lane_id == 0) {
    float amax_sc = 0.0f;
    float max_sc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
      float v = scales_sb[sb];
      float av = abs(v);
      if (av > amax_sc) {
        amax_sc = av;
        max_sc = v;
      }
    }
    device uint8_t* scales12 = block_addr + KQ_Q3_K_SCALES_OFFSET;
    for (int i = 0; i < 12; i++)
      scales12[i] = 0;

    float d_block = 0.0f;
    if (max_sc != 0.0f) {
      float iscale = -32.0f / max_sc;
      for (int sb = 0; sb < 16; sb++) {
        int l = kq_nearest_int(iscale * scales_sb[sb]);
        l = max(-32, min(31, l)) + 32; // biased [0, 63]
        if (sb < 8) {
          scales12[sb] = uint8_t(l & 0x0F);
        } else {
          scales12[sb - 8] |= uint8_t((l & 0x0F) << 4);
        }
        uint8_t lh = uint8_t((l >> 4) & 0x03);
        scales12[8 + (sb % 4)] |= uint8_t(lh << (2 * (sb / 4)));
      }
      d_block = 1.0f / iscale;
    }
    *(device half*)(block_addr + KQ_Q3_K_D_OFFSET) = half(d_block);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 5: re-quantize --
  int my_sb = int(lid) / 16;
  uint8_t sc_unsigned =
      kq_q3_k_unpack_scale(my_sb, block_addr + KQ_Q3_K_SCALES_OFFSET);
  int sc_signed = int(sc_unsigned) - 32;
  float d_fp16 = float(*(device half*)(block_addr + KQ_Q3_K_D_OFFSET));
  float d_eff = d_fp16 * float(sc_signed);

  uint8_t my_u3;
  if (d_eff == 0.0f) {
    my_u3 = 4;
  } else {
    int l = kq_nearest_int(Xs[lid] / d_eff);
    l = max(-4, min(3, l));
    my_u3 = uint8_t(l + 4);
  }
  L_tgm[lid] = my_u3;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 6: pack qs[64] + hmask[32] --
  if (lid < 64) {
    int outer_half = int(lid) / 32;
    int within_shift = int(lid) % 32;
    uint8_t byte = 0;
    for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
      int w_idx = outer_half * 128 + shift_idx * 32 + within_shift;
      uint8_t q2 = L_tgm[w_idx] & 0x03;
      byte |= uint8_t(q2 << (shift_idx * 2));
    }
    device uint8_t* qs = block_addr + KQ_Q3_K_QS_OFFSET;
    qs[lid] = byte;
  } else if (lid < 96) {
    int within_shift = int(lid) - 64;
    uint8_t byte = 0;
    for (int b = 0; b < 8; b++) {
      int outer_half = b / 4;
      int shift_idx = b % 4;
      int w_idx = outer_half * 128 + shift_idx * 32 + within_shift;
      uint8_t hbit = (L_tgm[w_idx] >> 2) & 0x01;
      byte |= uint8_t(hbit << b);
    }
    device uint8_t* hmask = block_addr + KQ_Q3_K_HMASK_OFFSET;
    hmask[within_shift] = byte;
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q3_k_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    const constant uint& K [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]) {
  static_assert(group_size == 256, "Q3_K requires group_size=256");
  static_assert(bits == 3, "Q3_K requires bits=3");
  threadgroup float Xs[256];
  threadgroup float QWs[256];
  threadgroup uint8_t L_tgm[256];
  threadgroup float scales_sb[16];
  threadgroup float scratch[16];
  kq_q3_k_quantize_impl<T>(
      w,
      out,
      num_blocks,
      imatrix,
      has_imatrix != 0,
      K,
      tg_id,
      lid,
      simd_id,
      lane_id,
      Xs,
      QWs,
      L_tgm,
      scales_sb,
      scratch);
}

template <typename T>
METAL_FUNC void kq_q2_k_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    const device float* imatrix,
    bool has_imatrix,
    uint K,
    uint tg_id,
    uint lid,
    uint simd_id,
    uint lane_id,
    threadgroup float* Xs,
    threadgroup float* QWs,
    threadgroup uint8_t* L_tgm,
    threadgroup float* scales_sb,
    threadgroup float* mins_sb,
    threadgroup float* sw_sb,
    threadgroup uint8_t* Ls,
    threadgroup uint8_t* Lm,
    threadgroup float* scratch) {
  if (tg_id >= num_blocks) {
    return;
  }

  device uint8_t* block_addr = out + tg_id * KQ_Q2_K_BLOCK_BYTES;
  const device T* x_global = w + tg_id * KQ_Q2_K_SUPERBLOCK;

  // -- Phase 1: Load Xs[256] --
  Xs[lid] = float(x_global[lid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  kq_compute_sigma2_av_x(Xs, scratch, lid, simd_id, lane_id, 1.0f);
  float sigma2 = scratch[8];
  float av_x = scratch[9];

  // -- Phase 2: weights --
  if (has_imatrix) {
    uint k_off = (tg_id * KQ_Q2_K_SUPERBLOCK) % K;
    QWs[lid] = imatrix[k_off + lid] * sqrt(sigma2 + Xs[lid] * Xs[lid]);
  } else {
    QWs[lid] = av_x + abs(Xs[lid]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 3: per-sub-block fit --
  if ((lid % 16) == 0) {
    int sb = int(lid) / 16;
    int sb_off = sb * 16;
    float sumw = 0.0f;
    for (int l = 0; l < 16; l++)
      sumw += QWs[sb_off + l];
    sw_sb[sb] = sumw;
    float the_min;
    float scale =
        kq_make_qkx3_quants<16>(&Xs[sb_off], &QWs[sb_off], 3, the_min);
    scales_sb[sb] = scale;
    mins_sb[sb] = the_min;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 4: super-scale + pack scales[16] --
  if (simd_id == 0 && lane_id == 0) {
    uint8_t Ls_local[16];
    uint8_t Lm_local[16];
    float d_block =
        kq_make_qp_quants<16>(&scales_sb[0], &sw_sb[0], 15, Ls_local);
    float m_block = kq_make_qp_quants<16>(&mins_sb[0], &sw_sb[0], 15, Lm_local);
    for (int i = 0; i < 16; i++) {
      Ls[i] = Ls_local[i];
      Lm[i] = Lm_local[i];
    }
    *(device half*)(block_addr + KQ_Q2_K_D_OFFSET) = half(d_block);
    *(device half*)(block_addr + KQ_Q2_K_DMIN_OFFSET) = half(m_block);

    device uint8_t* scales16 = block_addr + KQ_Q2_K_SCALES_OFFSET;
    for (int j = 0; j < 16; j++) {
      scales16[j] = uint8_t((Ls_local[j] & 0x0F) | ((Lm_local[j] & 0x0F) << 4));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 5: re-quantize --
  float d_wire = float(*(device half*)(block_addr + KQ_Q2_K_D_OFFSET));
  float dmin_wire = float(*(device half*)(block_addr + KQ_Q2_K_DMIN_OFFSET));
  device uint8_t* scales16 = block_addr + KQ_Q2_K_SCALES_OFFSET;
  int my_sb = int(lid) / 16;
  uint8_t sc_byte = scales16[my_sb];
  uint8_t sc = sc_byte & 0x0F;
  uint8_t mn = sc_byte >> 4;
  float d_eff = d_wire * float(sc);
  float m_eff = dmin_wire * float(mn);

  uint8_t my_L;
  if (d_eff == 0.0f) {
    my_L = 0;
  } else {
    int l = kq_nearest_int((Xs[lid] + m_eff) / d_eff);
    l = max(0, min(3, l));
    my_L = uint8_t(l);
  }
  L_tgm[lid] = my_L;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // -- Phase 6: pack qs[64] --
  if (lid < 64) {
    int outer_half = int(lid) / 32;
    int within_shift = int(lid) % 32;
    uint8_t byte = 0;
    for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
      int w_idx = outer_half * 128 + shift_idx * 32 + within_shift;
      uint8_t q2 = L_tgm[w_idx] & 0x03;
      byte |= uint8_t(q2 << (shift_idx * 2));
    }
    device uint8_t* qs = block_addr + KQ_Q2_K_QS_OFFSET;
    qs[lid] = byte;
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q2_k_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    const constant uint& K [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]) {
  static_assert(group_size == 256, "Q2_K requires group_size=256");
  static_assert(bits == 2, "Q2_K requires bits=2");
  threadgroup float Xs[256];
  threadgroup float QWs[256];
  threadgroup uint8_t L_tgm[256];
  threadgroup float scales_sb[16];
  threadgroup float mins_sb[16];
  threadgroup float sw_sb[16];
  threadgroup uint8_t Ls[16];
  threadgroup uint8_t Lm[16];
  threadgroup float scratch[16];
  kq_q2_k_quantize_impl<T>(
      w,
      out,
      num_blocks,
      imatrix,
      has_imatrix != 0,
      K,
      tg_id,
      lid,
      simd_id,
      lane_id,
      Xs,
      QWs,
      L_tgm,
      scales_sb,
      mins_sb,
      sw_sb,
      Ls,
      Lm,
      scratch);
}

template <typename T>
METAL_FUNC void kq_q4_0_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    uint gid) {
  if (gid >= num_blocks)
    return;
  const device T* x = w + gid * KQ_Q4_0_GROUP;
  device uint8_t* block_addr = out + gid * KQ_Q4_0_BLOCK_BYTES;

  float amax = 0.0f;
  for (int j = 0; j < KQ_Q4_0_GROUP; j++) {
    amax = max(amax, fabs(float(x[j])));
  }
  const float d = amax / 7.0f;
  const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

  *(device half*)(block_addr + KQ_Q4_0_D_OFFSET) = half(d);
  device uint8_t* qs = block_addr + KQ_Q4_0_QS_OFFSET;
  for (int j = 0; j < 16; j++) {
    float x0 = float(x[j]) * id;
    float x1 = float(x[j + 16]) * id;
    uint8_t q0 = uint8_t(clamp(round(x0) + 8.0f, 0.0f, 15.0f));
    uint8_t q1 = uint8_t(clamp(round(x1) + 8.0f, 0.0f, 15.0f));
    qs[j] = q0 | (q1 << 4);
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q4_0_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_Q4_0_GROUP, "Q4_0 requires group_size=32");
  static_assert(bits == 4, "Q4_0 requires bits=4");
  (void)imatrix;
  (void)has_imatrix;
  kq_q4_0_quantize_impl<T>(w, out, num_blocks, gid);
}

template <typename T>
METAL_FUNC void kq_q4_1_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    uint gid) {
  if (gid >= num_blocks)
    return;
  const device T* x = w + gid * KQ_Q4_1_GROUP;
  device uint8_t* block_addr = out + gid * KQ_Q4_1_BLOCK_BYTES;

  float vmin = float(x[0]);
  float vmax = float(x[0]);
  for (int j = 1; j < KQ_Q4_1_GROUP; j++) {
    float v = float(x[j]);
    vmin = min(vmin, v);
    vmax = max(vmax, v);
  }
  const float d = (vmax - vmin) / 15.0f;
  const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

  *(device half*)(block_addr + KQ_Q4_1_D_OFFSET) = half(d);
  *(device half*)(block_addr + KQ_Q4_1_M_OFFSET) = half(vmin);
  device uint8_t* qs = block_addr + KQ_Q4_1_QS_OFFSET;
  for (int j = 0; j < 16; j++) {
    float x0 = (float(x[j]) - vmin) * id;
    float x1 = (float(x[j + 16]) - vmin) * id;
    uint8_t q0 = uint8_t(clamp(round(x0), 0.0f, 15.0f));
    uint8_t q1 = uint8_t(clamp(round(x1), 0.0f, 15.0f));
    qs[j] = q0 | (q1 << 4);
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q4_1_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_Q4_1_GROUP, "Q4_1 requires group_size=32");
  static_assert(bits == 4, "Q4_1 requires bits=4");
  (void)imatrix;
  (void)has_imatrix;
  kq_q4_1_quantize_impl<T>(w, out, num_blocks, gid);
}

template <typename T>
METAL_FUNC void kq_q5_0_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    uint gid) {
  if (gid >= num_blocks)
    return;
  const device T* x = w + gid * KQ_Q5_0_GROUP;
  device uint8_t* block_addr = out + gid * KQ_Q5_0_BLOCK_BYTES;

  float amax = 0.0f;
  for (int j = 0; j < KQ_Q5_0_GROUP; j++) {
    amax = max(amax, fabs(float(x[j])));
  }
  const float d = amax / 15.0f;
  const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

  *(device half*)(block_addr + KQ_Q5_0_D_OFFSET) = half(d);
  device uint8_t* qh_p = block_addr + KQ_Q5_0_QH_OFFSET;
  device uint8_t* qs = block_addr + KQ_Q5_0_QS_OFFSET;

  uint32_t qh = 0;
  for (int j = 0; j < 16; j++) {
    float v0 = float(x[j]) * id;
    float v1 = float(x[j + 16]) * id;
    uint8_t q0 = uint8_t(clamp(round(v0) + 16.0f, 0.0f, 31.0f));
    uint8_t q1 = uint8_t(clamp(round(v1) + 16.0f, 0.0f, 31.0f));
    qs[j] = (q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4);
    qh |= (uint32_t(q0 >> 4) << j);
    qh |= (uint32_t(q1 >> 4) << (j + 16));
  }
  qh_p[0] = uint8_t(qh & 0xFF);
  qh_p[1] = uint8_t((qh >> 8) & 0xFF);
  qh_p[2] = uint8_t((qh >> 16) & 0xFF);
  qh_p[3] = uint8_t((qh >> 24) & 0xFF);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_0_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_Q5_0_GROUP, "Q5_0 requires group_size=32");
  static_assert(bits == 5, "Q5_0 requires bits=5");
  (void)imatrix;
  (void)has_imatrix;
  kq_q5_0_quantize_impl<T>(w, out, num_blocks, gid);
}

template <typename T>
METAL_FUNC void kq_q5_1_quantize_impl(
    const device T* w,
    device uint8_t* out,
    const constant uint& num_blocks,
    uint gid) {
  if (gid >= num_blocks)
    return;
  const device T* x = w + gid * KQ_Q5_1_GROUP;
  device uint8_t* block_addr = out + gid * KQ_Q5_1_BLOCK_BYTES;

  float vmin = float(x[0]);
  float vmax = float(x[0]);
  for (int j = 1; j < KQ_Q5_1_GROUP; j++) {
    float v = float(x[j]);
    vmin = min(vmin, v);
    vmax = max(vmax, v);
  }
  const float d = (vmax - vmin) / 31.0f;
  const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

  *(device half*)(block_addr + KQ_Q5_1_D_OFFSET) = half(d);
  *(device half*)(block_addr + KQ_Q5_1_M_OFFSET) = half(vmin);
  device uint8_t* qh_p = block_addr + KQ_Q5_1_QH_OFFSET;
  device uint8_t* qs = block_addr + KQ_Q5_1_QS_OFFSET;

  uint32_t qh = 0;
  for (int j = 0; j < 16; j++) {
    float v0 = (float(x[j]) - vmin) * id;
    float v1 = (float(x[j + 16]) - vmin) * id;
    uint8_t q0 = uint8_t(clamp(round(v0), 0.0f, 31.0f));
    uint8_t q1 = uint8_t(clamp(round(v1), 0.0f, 31.0f));
    qs[j] = (q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4);
    qh |= (uint32_t(q0 >> 4) << j);
    qh |= (uint32_t(q1 >> 4) << (j + 16));
  }
  qh_p[0] = uint8_t(qh & 0xFF);
  qh_p[1] = uint8_t((qh >> 8) & 0xFF);
  qh_p[2] = uint8_t((qh >> 16) & 0xFF);
  qh_p[3] = uint8_t((qh >> 24) & 0xFF);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_1_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    const constant uint& num_blocks [[buffer(2)]],
    const device float* imatrix [[buffer(3)]],
    const constant uint& has_imatrix [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_Q5_1_GROUP, "Q5_1 requires group_size=32");
  static_assert(bits == 5, "Q5_1 requires bits=5");
  (void)imatrix;
  (void)has_imatrix;
  kq_q5_1_quantize_impl<T>(w, out, num_blocks, gid);
}
