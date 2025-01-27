// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/utils.h"

#define MLX_MTL_CONST static constant constexpr const

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Naive unfold with dilation
///////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
[[kernel]] void naive_unfold_Nd(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MLXConvParams<N>* params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  int filter_size = params->C;
  for (short i = 0; i < N; i++)
    filter_size *= params->wS[i];

  int out_pixels = 1;
  for (short i = 0; i < N; i++)
    out_pixels *= params->oS[i];

  // Set out
  out += gid.z * filter_size + gid.y * (params->C);

  // Coordinates in input
  int is[N] = {0};

  // gid.z: N oS (Batch and row in unfolded output)
  // gid.y: wS (Filter location to unfold input)
  // gid.x: C (channel)

  int n = (gid.z) / out_pixels;
  int oS = (gid.z) % out_pixels;
  int wS = gid.y;

  bool valid = n < params->N;

  // Unroll dimensions
  for (int i = N - 1; i >= 0; --i) {
    int os_ = (oS % params->oS[i]);
    int ws_ = (wS % params->wS[i]);

    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;

    int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    int is_max = 1 + params->idil[i] * (params->iS[i] - 1);

    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);

    is[i] = is_ / params->idil[i];

    oS /= params->oS[i];
    wS /= params->wS[i];
  }

  if (valid) {
    size_t in_offset = n * params->in_strides[0];

    for (int i = 0; i < N; ++i) {
      in_offset += is[i] * params->in_strides[i + 1];
    }

    out[gid.x] = in[in_offset + gid.x];
  } else {
    out[gid.x] = T(0);
  }
}

// This kernel unfolds the input array of size (N, *spatial_dims, C)
// into an array of size (N x *spatial_dims, C x *kernel_dims).
template <typename T, int N>
[[kernel]] void naive_unfold_transpose_Nd(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MLXConvParams<N>* params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  int filter_size = params->C;
  for (short i = 0; i < N; i++)
    filter_size *= params->wS[i];

  int out_pixels = 1;
  for (short i = 0; i < N; i++)
    out_pixels *= params->oS[i];

  // Set out
  out += gid.z * filter_size + gid.x * (filter_size / params->C);

  // Coordinates in input
  int is[N] = {0};

  // gid.z: N oS (Batch and row in unfolded output)
  // gid.y: wS (Filter location to unfold input)
  // gid.x: C (channel)

  int n = (gid.z) / out_pixels;
  int oS = (gid.z) % out_pixels;
  int wS = gid.y;

  bool valid = n < params->N;

  // Unroll dimensions
  int kernel_stride = 1;
  for (int i = N - 1; i >= 0; --i) {
    int os_ = (oS % params->oS[i]);
    int ws_ = (wS % params->wS[i]);
    out += ws_ * kernel_stride;

    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;

    int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    int is_max = 1 + params->idil[i] * (params->iS[i] - 1);

    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);

    is[i] = is_ / params->idil[i];

    oS /= params->oS[i];
    wS /= params->wS[i];

    kernel_stride *= params->wS[i];
  }

  if (valid) {
    size_t in_offset = n * params->in_strides[0];

    for (int i = 0; i < N; ++i) {
      in_offset += is[i] * params->in_strides[i + 1];
    }

    out[0] = in[in_offset + gid.x];
  } else {
    out[0] = T(0);
  }
}

#define instantiate_naive_unfold_nd(name, itype, n)                            \
  template [[host_name("naive_unfold_nd_" #name "_" #n)]] [[kernel]] void      \
  naive_unfold_Nd(                                                             \
      const device itype* in [[buffer(0)]],                                    \
      device itype* out [[buffer(1)]],                                         \
      const constant MLXConvParams<n>* params [[buffer(2)]],                   \
      uint3 gid [[thread_position_in_grid]]);                                  \
  template                                                                     \
      [[host_name("naive_unfold_transpose_nd_" #name "_" #n)]] [[kernel]] void \
      naive_unfold_transpose_Nd(                                               \
          const device itype* in [[buffer(0)]],                                \
          device itype* out [[buffer(1)]],                                     \
          const constant MLXConvParams<n>* params [[buffer(2)]],               \
          uint3 gid [[thread_position_in_grid]]);

#define instantiate_naive_unfold_nd_dims(name, itype)                      \
  instantiate_naive_unfold_nd(name, itype, 1) instantiate_naive_unfold_nd( \
      name, itype, 2) instantiate_naive_unfold_nd(name, itype, 3)

instantiate_naive_unfold_nd_dims(float32, float);
instantiate_naive_unfold_nd_dims(float16, half);
instantiate_naive_unfold_nd_dims(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Slow and naive conv2d kernels
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const int BC = 16>
[[kernel]] void naive_conv_2d(
    const device T* in [[buffer(0)]],
    const device T* wt [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant MLXConvParams<2>& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)simd_gid;
  (void)simd_lid;

  out += tid.z * params.out_strides[0];
  in += tid.z * params.in_strides[0];

  int out_o = tid.y * BN * TN + lid.y * TN;
  int out_hw = tid.x * BM * TM + lid.x * TM;

  int out_h[TM];
  int out_w[TN];

  for (int m = 0; m < TM; ++m) {
    int mm = (out_hw + m);
    out_h[m] = mm / params.oS[1];
    out_w[m] = mm % params.oS[1];
  }

  T in_local[TM];
  T wt_local[TN];
  T out_local[TM * TN] = {T(0)};

  for (int h = 0; h < params.wS[0]; ++h) {
    for (int w = 0; w < params.wS[1]; ++w) {
      for (int c = 0; c < params.C; ++c) {
        // Local in
        for (int m = 0; m < TM; m++) {
          int i = out_h[m] * params.str[0] - params.pad[0] + h * params.kdil[0];
          int j = out_w[m] * params.str[1] - params.pad[1] + w * params.kdil[1];

          bool valid = i >= 0 && i < params.iS[0] && j >= 0 && j < params.iS[1];
          in_local[m] = valid
              ? in[i * params.in_strides[1] + j * params.in_strides[2] + c]
              : T(0);
        }

        // Load weight
        for (int n = 0; n < TN; ++n) {
          int o = out_o + n;
          wt_local[n] = o < params.O
              ? wt[o * params.wt_strides[0] + h * params.wt_strides[1] +
                   w * params.wt_strides[2] + c]
              : T(0);
        }

        // Accumulate
        for (int m = 0; m < TM; ++m) {
          for (int n = 0; n < TN; ++n) {
            out_local[m * TN + n] += in_local[m] * wt_local[n];
          }
        }
      }
    }
  }

  for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
      if (out_h[m] < params.oS[0] && out_w[m] < params.oS[1] &&
          (out_o + n) < params.O)
        out[out_h[m] * params.out_strides[1] +
            out_w[m] * params.out_strides[2] + out_o + n] =
            out_local[m * TN + n];
    }
  }
}

// Instantiations

#define instantiate_naive_conv_2d(name, itype, bm, bn, tm, tn)              \
  template [[host_name("naive_conv_2d_" #name "_bm" #bm "_bn" #bn "_tm" #tm \
                       "_tn" #tn)]] [[kernel]] void                         \
  naive_conv_2d<itype, bm, bn, tm, tn>(                                     \
      const device itype* in [[buffer(0)]],                                 \
      const device itype* wt [[buffer(1)]],                                 \
      device itype* out [[buffer(2)]],                                      \
      const constant MLXConvParams<2>& params [[buffer(3)]],                \
      uint3 tid [[threadgroup_position_in_grid]],                           \
      uint3 lid [[thread_position_in_threadgroup]],                         \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                     \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_naive_conv_2d_blocks(name, itype) \
  instantiate_naive_conv_2d(name, itype, 16, 8, 4, 4) \
      instantiate_naive_conv_2d(name, itype, 16, 8, 2, 4)

instantiate_naive_conv_2d_blocks(float32, float);
instantiate_naive_conv_2d_blocks(float16, half);
instantiate_naive_conv_2d_blocks(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Winograd kernels
///////////////////////////////////////////////////////////////////////////////

template <int M, int R, int S>
struct WinogradTransforms {};

template <>
struct WinogradTransforms<6, 3, 8> {
  MLX_MTL_CONST int OUT_TILE_SIZE = 6;
  MLX_MTL_CONST int FILTER_SIZE = 3;
  MLX_MTL_CONST int IN_TILE_SIZE = OUT_TILE_SIZE + FILTER_SIZE - 1;
  MLX_MTL_CONST int SIMD_MATRIX_SIZE = 8;
  MLX_MTL_CONST float in_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 0.50f, -0.50f, 2.00f, -2.00f, -1.00f},
      {-5.25f, 1.00f, 1.00f, 0.25f, 0.25f, 4.00f, 4.00f, 0.00f},
      {0.00f, -4.25f, 4.25f, -2.50f, 2.50f, -2.50f, 2.50f, 5.25f},
      {5.25f, -4.25f, -4.25f, -1.25f, -1.25f, -5.00f, -5.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 2.00f, -2.00f, 0.50f, -0.50f, -5.25f},
      {-1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 0.00f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  MLX_MTL_CONST float out_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
      {1.00f, -1.00f, 1.00f, -1.00f, 1.00f, -1.00f},
      {1.00f, 2.00f, 4.00f, 8.00f, 16.00f, 32.00f},
      {1.00f, -2.00f, 4.00f, -8.00f, 16.00f, -32.00f},
      {1.00f, 0.50f, 0.25f, 0.125f, 0.0625f, 0.03125f},
      {1.00f, -0.50f, 0.25f, -0.125f, 0.0625f, -0.03125f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  MLX_MTL_CONST float wt_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00, 0.00, 0.00},
      {-2.0 / 9.00, -2.0 / 9.00, -2.0 / 9.00},
      {-2.0 / 9.00, 2.0 / 9.00, -2.0 / 9.00},
      {1.0 / 90.0, 1.0 / 45.0, 2.0 / 45.0},
      {1.0 / 90.0, -1.0 / 45.0, 2.0 / 45.0},
      {32.0 / 45.0, 16.0 / 45.0, 8.0 / 45.0},
      {32.0 / 45.0, -16.0 / 45.0, 8.0 / 45.0},
      {0.00, 0.00, 1.00},
  };
};

constant constexpr const float WinogradTransforms<6, 3, 8>::wt_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::in_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::out_transform[8][8];

template <typename T, int BC = 32, int BO = 4, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(BO * 32)]] void
winograd_conv_2d_weight_transform(
    const device T* wt_in [[buffer(0)]],
    device T* wt_out [[buffer(1)]],
    const constant int& C [[buffer(2)]],
    const constant int& O [[buffer(3)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  using WGT = WinogradTransforms<M, R, 8>;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize G matrix
  simdgroup_matrix<float, 8, 8> G;
  G.thread_elements()[0] = WGT::wt_transform[sm][sn];
  G.thread_elements()[1] = WGT::wt_transform[sm][sn + 1];

  // Initialize Gt matrix
  simdgroup_matrix<float, 8, 8> Gt;
  Gt.thread_elements()[0] = WGT::wt_transform[sn][sm];
  Gt.thread_elements()[1] = WGT::wt_transform[sn + 1][sm];

  // Move to the correct output filter
  size_t ko = BO * tid + simd_group_id;
  wt_in += ko * R * R * C;

  // wt_out is stored transposed (A x A x C x O)
  short ohw_0 = sm * 8 + sn;
  short ohw_1 = sm * 8 + sn + 1;
  device T* wt_out_0 = wt_out + ohw_0 * C * O + ko;
  device T* wt_out_1 = wt_out + ohw_1 * C * O + ko;

  // Prepare shared memory
  threadgroup T Ws[BO][R][R][BC];

  // Loop over C
  for (int bc = 0; bc < C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int kh = 0; kh < R; ++kh) {
      for (int kw = 0; kw < R; ++kw) {
        for (int kc = simd_lane_id; kc < BC; kc += 32) {
          Ws[simd_group_id][kh][kw][kc] = wt_in[kh * R * C + kw * C + kc];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = 0; c < BC; ++c) {
      simdgroup_matrix<float, 8, 8> g;
      g.thread_elements()[0] =
          sm < R && sn < R ? Ws[simd_group_id][sm][sn][c] : T(0);
      g.thread_elements()[1] =
          sm < R && sn + 1 < R ? Ws[simd_group_id][sm][sn + 1][c] : T(0);

      simdgroup_matrix<float, 8, 8> g_out = (G * g) * Gt;
      wt_out_0[c * O] = static_cast<T>(g_out.thread_elements()[0]);
      wt_out_1[c * O] = static_cast<T>(g_out.thread_elements()[1]);
    }

    wt_in += BC;
    wt_out_0 += BC * O;
    wt_out_1 += BC * O;
  }
}

#define instantiate_winograd_conv_2d_weight_transform_base(name, itype, bc) \
  template [[host_name("winograd_conv_2d_weight_transform_" #name           \
                       "_bc" #bc)]] [[kernel]] void                         \
  winograd_conv_2d_weight_transform<itype, bc>(                             \
      const device itype* wt_in [[buffer(0)]],                              \
      device itype* wt_out [[buffer(1)]],                                   \
      const constant int& C [[buffer(2)]],                                  \
      const constant int& O [[buffer(3)]],                                  \
      uint tid [[threadgroup_position_in_grid]],                            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BC, int WM, int WN, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
winograd_conv_2d_input_transform(
    const device T* inp_in [[buffer(0)]],
    device T* inp_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int A = WGT::IN_TILE_SIZE;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize B matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::in_transform[sm][sn];
  B.thread_elements()[1] = WGT::in_transform[sm][sn + 1];

  // Initialize Bt matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::in_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::in_transform[sn + 1][sm];

  // Resolve input tile
  constexpr int TH = (A / WM);
  constexpr int TW = (A / WN);
  const int kh = TH * (simd_group_id / WN);
  const int kw = TW * (simd_group_id % WN);
  const int bh = M * tid.y + kh - params.pad[1];
  const int bw = M * tid.x + kw - params.pad[0];

  const bool is_edge_w_lo = bw < 0;
  const bool is_edge_h_lo = bh < 0;
  const bool is_edge_w_hi = bw + (TW - 1) >= params.iS[0];
  const bool is_edge_h_hi = bh + (TH - 1) >= params.iS[1];
  const bool is_edge =
      is_edge_w_lo || is_edge_h_lo || is_edge_w_hi || is_edge_h_hi;

  // Move to the correct input tile
  inp_in += tid.z * params.in_strides[0] + bh * int64_t(params.in_strides[1]) +
      bw * int64_t(params.in_strides[2]);

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      jump_in[h][w] = h * params.in_strides[1] + w * params.in_strides[2];
    }
  }

  // inp_out is stored interleaved (A x A x tiles x C)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  device T* inp_out_0 =
      inp_out + ohw_0 * N_TILES * params.C + tile_id * params.C;
  device T* inp_out_1 =
      inp_out + ohw_1 * N_TILES * params.C + tile_id * params.C;

  // Prepare shared memory
  threadgroup T Is[A][A][BC];

  // Loop over C
  for (int bc = 0; bc < params.C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        const device T* in_ptr = inp_in + jump_in[h][w];
        if (is_edge) {
          if (((bh + h) < 0 || (bh + h) >= params.iS[1]) ||
              ((bw + w) < 0 || (bw + w) >= params.iS[0])) {
            for (int c = simd_lane_id; c < BC; c += 32) {
              Is[kh + h][kw + w][c] = T(0);
            }
          } else {
            for (int c = simd_lane_id; c < BC; c += 32) {
              Is[kh + h][kw + w][c] = in_ptr[c];
            }
          }
        } else {
          for (int c = simd_lane_id; c < BC; c += 32) {
            Is[kh + h][kw + w][c] = in_ptr[c];
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BC; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> I;
      I.thread_elements()[0] = Is[sm][sn][c];
      I.thread_elements()[1] = Is[sm][sn + 1][c];

      simdgroup_matrix<float, 8, 8> I_out = (Bt * I) * B;
      inp_out_0[c] = static_cast<T>(I_out.thread_elements()[0]);
      inp_out_1[c] = static_cast<T>(I_out.thread_elements()[1]);
    }

    inp_in += BC;
    inp_out_0 += BC;
    inp_out_1 += BC;
  }
}

#define instantiate_winograd_conv_2d_input_transform(name, itype, bc) \
  template [[host_name("winograd_conv_2d_input_transform_" #name      \
                       "_bc" #bc)]] [[kernel]] void                   \
  winograd_conv_2d_input_transform<itype, bc, 2, 2>(                  \
      const device itype* inp_in [[buffer(0)]],                       \
      device itype* inp_out [[buffer(1)]],                            \
      const constant MLXConvParams<2>& params [[buffer(2)]],          \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint3 lid [[thread_position_in_threadgroup]],                   \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                   \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],          \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BO, int WM, int WN, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
winograd_conv_2d_output_transform(
    const device T* out_in [[buffer(0)]],
    device T* out_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize A matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::out_transform[sm][sn];
  B.thread_elements()[1] = WGT::out_transform[sm][sn + 1];

  // Initialize At matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::out_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::out_transform[sn + 1][sm];

  // Out_in comes in shape (A x A x tiles x O)
  // We do transform and then write out to out_out in shape (N, H, W, O)

  // Resolve output tile
  constexpr int TH = (M / WM);
  constexpr int TW = (M / WN);
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  out_out += tid.z * params.out_strides[0] + bh * params.out_strides[1] +
      bw * params.out_strides[2];

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      bool valid = ((bh + h) < params.oS[0]) && ((bw + w) < params.oS[1]);
      jump_in[h][w] =
          valid ? h * params.out_strides[1] + w * params.out_strides[2] : -1;
    }
  }

  // out_in is stored interleaved (A x A x tiles x O)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  const device T* out_in_0 =
      out_in + ohw_0 * N_TILES * params.O + tile_id * params.O;
  const device T* out_in_1 =
      out_in + ohw_1 * N_TILES * params.O + tile_id * params.O;

  // Prepare shared memory
  threadgroup T Os[M][M][BO];

  // Loop over O
  for (int bo = 0; bo < params.O; bo += BO) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BO; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> O_mat;
      O_mat.thread_elements()[0] = out_in_0[c];
      O_mat.thread_elements()[1] = out_in_1[c];

      simdgroup_matrix<float, 8, 8> O_out = (Bt * (O_mat * B));
      if ((sm < M) && (sn < M)) {
        Os[sm][sn][c] = static_cast<T>(O_out.thread_elements()[0]);
      }
      if ((sm < M) && ((sn + 1) < M)) {
        Os[sm][sn + 1][c] = static_cast<T>(O_out.thread_elements()[1]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read out from shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        if (jump_in[h][w] >= 0) {
          device T* out_ptr = out_out + jump_in[h][w];
          for (int c = simd_lane_id; c < BO; c += 32) {
            out_ptr[c] = Os[kh + h][kw + w][c];
          }
        }
      }
    }

    out_out += BO;
    out_in_0 += BO;
    out_in_1 += BO;
  }
}

#define instantiate_winograd_conv_2d_output_transform(name, itype, bo) \
  template [[host_name("winograd_conv_2d_output_transform_" #name      \
                       "_bo" #bo)]] [[kernel]] void                    \
  winograd_conv_2d_output_transform<itype, bo, 2, 2>(                  \
      const device itype* out_in [[buffer(0)]],                        \
      device itype* out_out [[buffer(1)]],                             \
      const constant MLXConvParams<2>& params [[buffer(2)]],           \
      uint3 tid [[threadgroup_position_in_grid]],                      \
      uint3 lid [[thread_position_in_threadgroup]],                    \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                    \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_winograd_conv_2d(name, itype)                     \
  instantiate_winograd_conv_2d_weight_transform_base(name, itype, 32) \
  instantiate_winograd_conv_2d_input_transform(name, itype, 32)       \
  instantiate_winograd_conv_2d_output_transform(name, itype, 32) // clang-format on

// clang-format off
instantiate_winograd_conv_2d(float32, float);
instantiate_winograd_conv_2d(bfloat16, bfloat16_t);
instantiate_winograd_conv_2d(float16, half); // clang-format on

#include "mlx/backend/metal/kernels/steel/attn/mma.h"

template <typename T, int WM = 4, int WN = 1, typename AccumType = float>
[[kernel]] void winograd_fused(
    const device T* input [[buffer(0)]],
    const device T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    const constant MLXConvParams<2>& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]) {
  using namespace mlx::steel;

  // Winograd F(n x n, r x r)
  // n x n output window
  constexpr short FN = 2;
  // r x r filter size
  constexpr short FR = 3;
  // a x a input window, a = n + r - 1
  constexpr short FA = 4;

  constexpr short kFragSize = 8; // MMA frag size

  constexpr short BT = 8; // Tile block size
  constexpr short BO = 8; // Output channel block size
  constexpr short BC = 8; // Input channel block size

  // clang-format off
  static_assert(BT % (1 * kFragSize) == 0 && 
                BO % (1 * kFragSize) == 0 && 
                BC % kFragSize == 0,
                "Matmuls sizes must be compatible with fragments");
  // clang-format on

  // Prepare for matmul

  // Warp tile sizes for matmul
  constexpr short TM = (FA * FA * BT) / (WM * kFragSize);
  constexpr short TN = (BO) / (WN * kFragSize);
  constexpr short TK = (BC) / (kFragSize);

  // Warp primitives
  using MMAFrag_inp_t = BaseMMAFrag<T, kFragSize, kFragSize>;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tiles sizes for matmul
  MMATile<T, 1, TK, MMAFrag_inp_t> Itile;
  MMATile<T, TK, TN, MMAFrag_inp_t> Wtile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Otile[4];

  for (int im = 0; im < 4; im++) {
    Otile[im].clear();
  }

  // Threadgroup memory for Weights and Inputs
  constexpr short BS = BT > BO ? BT : BO;
  threadgroup T Wt[FA * FA * BC * BO];
  threadgroup T It[FA * FA * BS * BS];

  // Get thread position in tile
  short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  static_assert(FA * FA * BT == 32 * WM * WN, "Each thread loads one pixel.");
  const int thr_idx = simd_group_id * 32 + simd_lane_id;
  const int thr_t = thr_idx / (FA * FA);
  const int thr_hw = thr_idx % (FA * FA);
  const int thr_h = thr_hw / FA;
  const int thr_w = thr_hw % FA;

  // Get batch, tile, and output idx for warp
  const int b_idx = tid.z;
  const int t_idx = BT * tid.y + thr_t;
  const int o_idx = BO * tid.x + thr_t;

  // Divide tile into h, w tile
  uniform<int> oHu = make_uniform(params.oS[0]);
  uniform<int> oWu = make_uniform(params.oS[1]);
  uniform<int> tHu = (oHu + make_uniform(FN - 1)) / make_uniform(FN);
  uniform<int> tWu = (oWu + make_uniform(FN - 1)) / make_uniform(FN);

  const int oH_idx = FN * (t_idx / tWu);
  const int oW_idx = FN * (t_idx % tWu);
  const int iH_idx = oH_idx + thr_h - params.pad[0];
  const int iW_idx = oW_idx + thr_w - params.pad[1];

  // Move to correct location

  // clang-format off
  input +=  b_idx * int64_t(params.in_strides[0]) + // N
           iH_idx * int64_t(params.in_strides[1]) + // H
           iW_idx * int64_t(params.in_strides[2]);  // W

  // output +=  b_idx * int64_t(params.out_strides[0]) + // N
  //           oH_idx * int64_t(params.out_strides[1]) + // H
  //           oW_idx * int64_t(params.out_strides[2]) + // W
  //           o_idx;                                    // C

  weight += o_idx * params.wt_strides[0] + // O
            thr_h * params.wt_strides[1] + // H 
            thr_w * params.wt_strides[2];  // W
  // clang-format on

  // Do edge check prep for input
  const bool is_edge_w_lo = iH_idx < 0;
  const bool is_edge_h_lo = iW_idx < 0;
  const bool is_edge_w_hi = iH_idx >= params.iS[0];
  const bool is_edge_h_hi = iW_idx >= params.iS[1];
  const bool is_edge =
      is_edge_w_lo || is_edge_h_lo || is_edge_w_hi || is_edge_h_hi;

  // Iterate over C
  for (int c = 0; c < params.C; c += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Load weight

#define tmp_load_wt_idx(o, h, w, c) h* FA* BC* BO + w* BC* BO + c* BO + o
#define tmp_load_in_idx(t, h, w, c) h* FA* BS* BC + w* BS* BC + t* BC + c

#define tmp_trns_wt_idx(o, h, w, c) h* FA* BC* BO + w* BC* BO + c* BO + o
#define tmp_trns_in_idx(t, h, w, c) h* FA* BS* BC + w* BS* BC + t* BC + c

    if (thr_h < FR && thr_w < FR && thr_t < BO) {
      for (int ic = 0; ic < BC; ic++) {
        Wt[tmp_load_wt_idx(thr_t, thr_h, thr_w, ic)] = weight[c + ic];
      }
    }

    // Load input
    if (is_edge) {
      for (int ic = 0; ic < BC; ic++) {
        It[tmp_load_in_idx(thr_t, thr_h, thr_w, ic)] = T(0);
      }
    } else {
      for (int ic = 0; ic < BC; ic++) {
        It[tmp_load_in_idx(thr_t, thr_h, thr_w, ic)] = input[c + ic];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transform weight
    if (lid.z == 0) {
      const short ig = simd_group_id * 32 + simd_lane_id;
      const short ic = lid.y;
      const short io = lid.x;

      T tmp_0[4][4];
      T tmp_1[4][4];

      for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
          tmp_0[ii][jj] = Wt[tmp_load_wt_idx(io, ii, jj, ic)];
        }
      }

      //////////////////////////////////////////////

      tmp_1[0][0] = tmp_0[0][0];
      tmp_1[0][1] = tmp_0[0][1];
      tmp_1[0][2] = tmp_0[0][2];

      tmp_1[1][0] = T(0.5) * (tmp_0[0][0] + tmp_0[1][0] + tmp_0[2][0]);
      tmp_1[1][1] = T(0.5) * (tmp_0[0][1] + tmp_0[1][1] + tmp_0[2][1]);
      tmp_1[1][2] = T(0.5) * (tmp_0[0][2] + tmp_0[1][2] + tmp_0[2][2]);

      tmp_1[2][0] = tmp_1[1][0] - tmp_0[1][0];
      tmp_1[2][1] = tmp_1[1][1] - tmp_0[1][1];
      tmp_1[2][2] = tmp_1[1][2] - tmp_0[1][2];

      tmp_1[3][0] = tmp_0[2][0];
      tmp_1[3][1] = tmp_0[2][1];
      tmp_1[3][2] = tmp_0[2][2];

      //////////////////////////////////////////////
      tmp_0[0][0] = tmp_1[0][0];
      tmp_0[1][0] = tmp_1[1][0];
      tmp_0[2][0] = tmp_1[2][0];
      tmp_0[3][0] = tmp_1[3][0];

      tmp_0[0][1] = T(0.5) * (tmp_1[0][0] + tmp_1[0][1] + tmp_1[0][2]);
      tmp_0[1][1] = T(0.5) * (tmp_1[1][0] + tmp_1[1][1] + tmp_1[1][2]);
      tmp_0[2][1] = T(0.5) * (tmp_1[2][0] + tmp_1[2][1] + tmp_1[2][2]);
      tmp_0[3][1] = T(0.5) * (tmp_1[3][0] + tmp_1[3][1] + tmp_1[3][2]);

      tmp_0[0][2] = tmp_0[0][1] - tmp_1[0][1];
      tmp_0[1][2] = tmp_0[1][1] - tmp_1[1][1];
      tmp_0[2][2] = tmp_0[2][1] - tmp_1[2][1];
      tmp_0[3][2] = tmp_0[3][1] - tmp_1[3][1];

      tmp_0[0][3] = tmp_1[0][2];
      tmp_0[1][3] = tmp_1[1][2];
      tmp_0[2][3] = tmp_1[2][2];
      tmp_0[3][3] = tmp_1[3][2];

      for (int ii = 0; ii < 4; ++ii) {
        for (int jj = 0; jj < 4; ++jj) {
          Wt[tmp_trns_wt_idx(io, ii, jj, ic)] = tmp_0[ii][jj];
        }
      }
    }

    // Transform input
    else { // (simd_group_id >= 2)
      const short ig = (simd_group_id - 2) * 32 + simd_lane_id;
      const short it = lid.y;
      const short ic = lid.x;

      T tmp_0[4][4];
      T tmp_1[4][4];

      for (int ii = 0; ii < 4; ++ii) {
        for (int jj = 0; jj < 4; ++jj) {
          tmp_0[ii][jj] = It[tmp_load_in_idx(it, ii, jj, ic)];
        }
      }

      //////////////////////////////////////////////

      tmp_1[0][0] = tmp_0[0][0] - tmp_0[2][0];
      tmp_1[0][1] = tmp_0[0][1] - tmp_0[2][1];
      tmp_1[0][2] = tmp_0[0][2] - tmp_0[2][2];
      tmp_1[0][3] = tmp_0[0][3] - tmp_0[2][3];

      tmp_1[1][0] = tmp_0[1][0] + tmp_0[2][0];
      tmp_1[1][1] = tmp_0[1][1] + tmp_0[2][1];
      tmp_1[1][2] = tmp_0[1][2] + tmp_0[2][2];
      tmp_1[1][3] = tmp_0[1][3] + tmp_0[2][3];

      tmp_1[2][0] = tmp_0[2][0] - tmp_0[1][0];
      tmp_1[2][1] = tmp_0[2][1] - tmp_0[1][1];
      tmp_1[2][2] = tmp_0[2][2] - tmp_0[1][2];
      tmp_1[2][3] = tmp_0[2][3] - tmp_0[1][3];

      tmp_1[3][0] = tmp_0[1][0] - tmp_0[3][0];
      tmp_1[3][1] = tmp_0[1][1] - tmp_0[3][1];
      tmp_1[3][2] = tmp_0[1][2] - tmp_0[3][2];
      tmp_1[3][3] = tmp_0[1][3] - tmp_0[3][3];

      //////////////////////////////////////////////
      tmp_0[0][0] = tmp_1[0][0] - tmp_1[0][2];
      tmp_0[1][0] = tmp_1[1][0] - tmp_1[1][2];
      tmp_0[2][0] = tmp_1[2][0] - tmp_1[2][2];
      tmp_0[3][0] = tmp_1[3][0] - tmp_1[3][2];

      tmp_0[0][1] = tmp_1[0][1] + tmp_1[0][2];
      tmp_0[1][1] = tmp_1[1][1] + tmp_1[1][2];
      tmp_0[2][1] = tmp_1[2][1] + tmp_1[2][2];
      tmp_0[3][1] = tmp_1[3][1] + tmp_1[3][2];

      tmp_0[0][2] = tmp_1[0][2] - tmp_1[0][1];
      tmp_0[1][2] = tmp_1[1][2] - tmp_1[1][1];
      tmp_0[2][2] = tmp_1[2][2] - tmp_1[2][1];
      tmp_0[3][2] = tmp_1[3][2] - tmp_1[3][1];

      tmp_0[0][3] = tmp_1[0][1] - tmp_1[0][3];
      tmp_0[1][3] = tmp_1[1][1] - tmp_1[1][3];
      tmp_0[2][3] = tmp_1[2][1] - tmp_1[2][3];
      tmp_0[3][3] = tmp_1[3][1] - tmp_1[3][3];

      for (int ii = 0; ii < 4; ++ii) {
        for (int jj = 0; jj < 4; ++jj) {
          It[tmp_trns_in_idx(it, ii, jj, ic)] = tmp_0[ii][jj];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do matmul
    for (int im = 0; im < 4; im++) {
      simdgroup_barrier(mem_flags::mem_none);
      Itile.template load<T, 1, 1, BS, 1>(
          &It[simd_group_id * FA * BS * BS + im * BS * BS + sm * BS + sn]);
      simdgroup_barrier(mem_flags::mem_none);
      Wtile.template load<T, 1, 1, BO, 1>(
          &Wt[simd_group_id * FA * BC * BO + im * BC * BO + sm * BO + sn]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Otile[im], Itile, Wtile, Otile[im]);
    }
  }

  // Transform and write output
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int im = 0; im < 4; im++) {
    Otile[im].template store<T, 1, 1, BS, 1>(
        &It[simd_group_id * FA * BS * BS + im * BS * BS + sm * BS + sn]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid.z == 0) {
    const short it = lid.y;
    const short io = lid.x;

    T tmp_0[4][4];
    T tmp_1[2][4];
    T tmp_2[2][2];

    for (int ii = 0; ii < 4; ++ii) {
      for (int jj = 0; jj < 4; ++jj) {
        tmp_0[ii][jj] = It[tmp_trns_in_idx(it, ii, jj, io)];
      }
    }

    tmp_1[0][0] = tmp_0[0][0] + tmp_0[1][0] + tmp_0[2][0];
    tmp_1[0][1] = tmp_0[0][1] + tmp_0[1][1] + tmp_0[2][1];
    tmp_1[0][2] = tmp_0[0][2] + tmp_0[1][2] + tmp_0[2][2];
    tmp_1[0][3] = tmp_0[0][3] + tmp_0[1][3] + tmp_0[2][3];

    tmp_1[1][0] = tmp_0[1][0] - tmp_0[2][0] - tmp_0[3][0];
    tmp_1[1][1] = tmp_0[1][1] - tmp_0[2][1] - tmp_0[3][1];
    tmp_1[1][2] = tmp_0[1][2] - tmp_0[2][2] - tmp_0[3][2];
    tmp_1[1][3] = tmp_0[1][3] - tmp_0[2][3] - tmp_0[3][3];

    tmp_2[0][0] = tmp_1[0][0] + tmp_1[0][1] + tmp_1[0][2];
    tmp_2[1][0] = tmp_1[1][0] + tmp_1[1][1] + tmp_1[1][2];

    tmp_2[0][1] = tmp_1[0][1] - tmp_1[0][2] - tmp_1[0][3];
    tmp_2[1][1] = tmp_1[1][1] - tmp_1[1][2] - tmp_1[1][3];

    const int oH_i = FN * ((BT * tid.y + it) / tWu);
    const int oW_i = FN * ((BT * tid.y + it) % tWu);

    // clang-format off
    output += b_idx * int64_t(params.out_strides[0]) + // N
              oH_i * int64_t(params.out_strides[1]) +  // H
              oW_i * int64_t(params.out_strides[2]) +  // W
              BO * tid.x;                     // C

    // clang-format on

    output[0 * params.out_strides[1] + 0 * params.out_strides[2] + io] =
        tmp_2[0][0];
    output[0 * params.out_strides[1] + 1 * params.out_strides[2] + io] =
        tmp_2[0][1];
    output[1 * params.out_strides[1] + 0 * params.out_strides[2] + io] =
        tmp_2[1][0];
    output[1 * params.out_strides[1] + 1 * params.out_strides[2] + io] =
        tmp_2[1][1];
  }
}

// clang-format off
#define instantiate_winograd_conv_2d_fused(name, itype)    \
  template [[host_name("winograd_conv_2d_fused_" #name)]]              \
  [[kernel]] void winograd_fused<itype>(                  \
      const device itype* input [[buffer(0)]],                        \
      const device itype* weight [[buffer(1)]],                        \
      device itype* output [[buffer(2)]],                             \
      const constant MLXConvParams<2>& params [[buffer(3)]],           \
      uint3 tid [[threadgroup_position_in_grid]],                      \
      uint3 lid [[thread_position_in_threadgroup]],                    \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                    \
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],           \
      ushort simd_lane_id [[thread_index_in_simdgroup]]);

instantiate_winograd_conv_2d_fused(float32, float);
// instantiate_winograd_conv_2d_fused(float16, float16_t);
// clang-format on