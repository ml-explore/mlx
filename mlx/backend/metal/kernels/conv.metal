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
/// Depthwise convolution kernels
///////////////////////////////////////////////////////////////////////////////

constant int ker_h [[function_constant(00)]];
constant int ker_w [[function_constant(01)]];
constant int str_h [[function_constant(10)]];
constant int str_w [[function_constant(11)]];
constant int tgp_h [[function_constant(100)]];
constant int tgp_w [[function_constant(101)]];
constant bool do_flip [[function_constant(200)]];

constant int span_h = tgp_h * str_h + ker_h - 1;
constant int span_w = tgp_w * str_w + ker_w - 1;
constant int span_hw = span_h * span_w;

template <typename T>
[[kernel]] void depthwise_conv_2d(
    const device T* in [[buffer(0)]],
    const device T* wt [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant MLXConvParams<2>& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int tc = 8;
  constexpr int tw = 8;
  constexpr int th = 4;

  constexpr int c_per_thr = 8;

  constexpr int TGH = th * 2 + 6;
  constexpr int TGW = tw * 2 + 6;
  constexpr int TGC = tc;

  threadgroup T ins[TGH * TGW * TGC];

  const int n_tgblocks_h = params.oS[0] / th;
  const int n = tid.z / n_tgblocks_h;
  const int tghid = tid.z % n_tgblocks_h;
  const int oh = tghid * th + lid.z;
  const int ow = gid.y;
  const int c = gid.x;

  in += n * params.in_strides[0];

  // Load in
  {
    constexpr int n_threads = th * tw * tc;
    const int tg_oh = (tghid * th) * str_h - params.pad[0];
    const int tg_ow = (tid.y * tw) * str_w - params.pad[1];
    const int tg_c = tid.x * tc;

    const int thread_idx = simd_gid * 32 + simd_lid;
    constexpr int thr_per_hw = tc / c_per_thr;
    constexpr int hw_per_group = n_threads / thr_per_hw;

    const int thr_c = thread_idx % thr_per_hw;
    const int thr_hw = thread_idx / thr_per_hw;

    for (int hw = thr_hw; hw < span_hw; hw += hw_per_group) {
      const int h = hw / span_w;
      const int w = hw % span_w;

      const int ih = tg_oh + h;
      const int iw = tg_ow + w;

      const int in_s_offset = h * span_w * TGC + w * TGC;

      if (ih >= 0 && ih < params.iS[0] && iw >= 0 && iw < params.iS[1]) {
        const auto in_load =
            in + ih * params.in_strides[1] + iw * params.in_strides[2] + tg_c;

        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] =
              in_load[c_per_thr * thr_c + cc];
        }
      } else {
        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] = T(0);
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  wt += c * params.wt_strides[0];

  const auto ins_ptr =
      &ins[lid.z * str_h * span_w * TGC + lid.y * str_w * TGC + lid.x];
  float o = 0.;
  for (int h = 0; h < ker_h; ++h) {
    for (int w = 0; w < ker_w; ++w) {
      int wt_h = h;
      int wt_w = w;
      if (do_flip) {
        wt_h = ker_h - h - 1;
        wt_w = ker_w - w - 1;
      }
      auto inv = ins_ptr[h * span_w * TGC + w * TGC];
      auto wtv = wt[wt_h * ker_w + wt_w];
      o += inv * wtv;
    }
  }
  threadgroup_barrier(mem_flags::mem_none);

  out += n * params.out_strides[0] + oh * params.out_strides[1] +
      ow * params.out_strides[2];
  out[c] = static_cast<T>(o);
}

#define instantiate_depthconv2d(iname, itype) \
  instantiate_kernel("depthwise_conv_2d_" #iname, depthwise_conv_2d, itype)

instantiate_depthconv2d(float32, float);
instantiate_depthconv2d(float16, half);
instantiate_depthconv2d(bfloat16, bfloat16_t);

template <typename T, typename IdxT>
[[kernel]] void depthwise_conv_1d(
    const device T* in [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const IdxT strides[3],
    constant const int& kernel_size,
    uint3 tid [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  out += (tid.z * static_cast<IdxT>(grid_dim.y) + tid.y) * grid_dim.x + tid.x;
  in += tid.z * strides[0] + tid.y * strides[1] + tid.x * strides[2];
  w += tid.x * kernel_size;

  float acc = 0.0;
  for (int i = 0; i < kernel_size; ++i) {
    acc += static_cast<float>(in[0]) * w[i];
    in += strides[1];
  }
  *out = static_cast<T>(acc);
}

#define instantiate_depthconv1d(iname, itype)                         \
  instantiate_kernel(                                                 \
      "depthwise_conv_1d_" #iname, depthwise_conv_1d, itype, int32_t) \
      instantiate_kernel(                                             \
          "depthwise_conv_1d_" #iname "_large",                       \
          depthwise_conv_1d,                                          \
          itype,                                                      \
          int64_t)

instantiate_depthconv1d(float32, float);
instantiate_depthconv1d(float16, half);
instantiate_depthconv1d(bfloat16, bfloat16_t);

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
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  inp_in += tid.z * params.in_strides[0] + bh * params.in_strides[1] +
      bw * params.in_strides[2];

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
        for (int c = simd_lane_id; c < BC; c += 32) {
          Is[kh + h][kw + w][c] = in_ptr[c];
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
