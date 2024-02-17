// Copyright © 2023-2024 Apple Inc.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/bf16.h"

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
  for(short i = 0; i < N; i++) filter_size *= params->wS[i];

  int out_pixels = 1;
  for(short i = 0; i < N; i++) out_pixels *= params->oS[i];

  // Set out 
  out += gid.z * filter_size + gid.y * (params->C);

  // Corrdinates in input
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

    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[0] == 0);

    is[i] = is_ / params->idil[i];

    oS /= params->oS[i];
    wS /= params->wS[i];
  }

  if(valid) {
    size_t in_offset = n * params->in_strides[0];

    for(int i = 0; i < N; ++i) {
      in_offset += is[i] * params->in_strides[i + 1];
    }

    out[gid.x] = in[in_offset + gid.x];
  } else {
    out[gid.x] = T(0);
  }

}

#define instantiate_naive_unfold_nd(name, itype, n) \
  template [[host_name("naive_unfold_nd_" #name "_" #n)]] \
  [[kernel]] void naive_unfold_Nd( \
      const device itype* in [[buffer(0)]], \
      device itype* out [[buffer(1)]], \
      const constant MLXConvParams<n>* params [[buffer(2)]], \
      uint3 gid [[thread_position_in_grid]]);

#define instantiate_naive_unfold_nd_dims(name, itype) \
  instantiate_naive_unfold_nd(name, itype, 1) \
  instantiate_naive_unfold_nd(name, itype, 2) \
  instantiate_naive_unfold_nd(name, itype, 3)

instantiate_naive_unfold_nd_dims(float32, float);
instantiate_naive_unfold_nd_dims(float16, half);
instantiate_naive_unfold_nd_dims(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Slow and naive conv2d kernels
///////////////////////////////////////////////////////////////////////////////

template <typename T, 
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

  for(int m = 0; m < TM; ++m) {
    int mm = (out_hw + m);
    out_h[m] = mm / params.oS[1];
    out_w[m] = mm % params.oS[1];
  }


  T in_local[TM];
  T wt_local[TN];
  T out_local[TM * TN] = {T(0)};

  for(int h = 0; h < params.wS[0]; ++h) {
    for(int w = 0; w < params.wS[1]; ++w) {
      for(int c = 0; c < params.C; ++c) {

        // Local in
        for(int m = 0; m < TM; m++) {
          int i = out_h[m] * params.str[0] - params.pad[0] + h * params.kdil[0];
          int j = out_w[m] * params.str[1] - params.pad[1] + w * params.kdil[1];

          bool valid = i >= 0 && i < params.iS[0] && j >= 0 && j < params.iS[1];
          in_local[m] = valid ? in[i * params.in_strides[1] + j * params.in_strides[2] + c] : T(0);
        }

        // Load weight
        for (int n = 0; n < TN; ++n) {
          int o = out_o + n;
          wt_local[n] = o < params.O ? wt[o * params.wt_strides[0] + 
                                          h * params.wt_strides[1] + 
                                          w * params.wt_strides[2] + c] : T(0);
        }

        // Accumulate
        for(int m = 0; m < TM; ++m) {
          for(int n = 0; n < TN; ++n) {
            out_local[m * TN + n] += in_local[m] * wt_local[n];
          }
        }

      }
    }
  }

  for(int m = 0; m < TM; ++m) {
    for(int n = 0; n < TN; ++n) {
      if(out_h[m] < params.oS[0] && out_w[m] < params.oS[1] && (out_o + n) < params.O)
      out[out_h[m] * params.out_strides[1] +
          out_w[m] * params.out_strides[2] + out_o + n] = out_local[m * TN + n];
    }
  }

}

// Instantiations

#define instantiate_naive_conv_2d(name, itype, bm, bn, tm, tn) \
  template [[host_name("naive_conv_2d_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn)]] \
  [[kernel]] void naive_conv_2d<itype, bm, bn, tm, tn>( \
      const device itype* in [[buffer(0)]], \
      const device itype* wt [[buffer(1)]], \
      device itype* out [[buffer(2)]], \
      const constant MLXConvParams<2>& params [[buffer(3)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
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
struct WinogradTransforms {

};

template <>
struct WinogradTransforms<6, 3, 8> {
  MLX_MTL_CONST int OUT_TILE_SIZE = 6;
  MLX_MTL_CONST int FILTER_SIZE = 3;
  MLX_MTL_CONST int IN_TILE_SIZE = OUT_TILE_SIZE + FILTER_SIZE - 1;
  MLX_MTL_CONST int SIMD_MATRIX_SIZE = 8;
  MLX_MTL_CONST float in_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
    { 1.00f,  0.00f,  0.00f,  0.00f,  0.00f,  0.00f,  0.00f,  0.00f},
    { 0.00f,  1.00f, -1.00f,  0.50f, -0.50f,  2.00f, -2.00f, -1.00f},
    {-5.25f,  1.00f,  1.00f,  0.25f,  0.25f,  4.00f,  4.00f,  0.00f},
    { 0.00f, -4.25f,  4.25f, -2.50f,  2.50f, -2.50f,  2.50f,  5.25f},
    { 5.25f, -4.25f, -4.25f, -1.25f, -1.25f, -5.00f, -5.00f,  0.00f},
    { 0.00f,  1.00f, -1.00f,  2.00f, -2.00f,  0.50f, -0.50f, -5.25f},
    {-1.00f,  1.00f,  1.00f,  1.00f,  1.00f,  1.00f,  1.00f,  0.00f},
    { 0.00f,  0.00f,  0.00f,  0.00f,  0.00f,  0.00f,  0.00f,  1.00f},
  };

  MLX_MTL_CONST float out_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
    { 1.00f,  0.00f,  0.00f,   0.00f,    0.00f,    0.00f},
    { 1.00f,  1.00f,  1.00f,   1.00f,    1.00f,    1.00f},
    { 1.00f, -1.00f,  1.00f,  -1.00f,    1.00f,   -1.00f},
    { 1.00f,  2.00f,  4.00f,   8.00f,   16.00f,   32.00f},
    { 1.00f, -2.00f,  4.00f,  -8.00f,   16.00f,  -32.00f},
    { 1.00f,  0.50f,  0.25f,  0.125f,   0.0625f,   0.03125f},
    { 1.00f, -0.50f,  0.25f, -0.125f,   0.0625f,  -0.03125f},
    { 0.00f,  0.00f,  0.00f,  0.00f,      0.00f,   1.00f},
  };

  MLX_MTL_CONST float wt_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
    {      1.00,       0.00,       0.00},
    { -2.0/9.00,  -2.0/9.00,  -2.0/9.00},
    { -2.0/9.00,   2.0/9.00,  -2.0/9.00},
    {  1.0/90.0,   1.0/45.0,   2.0/45.0},
    {  1.0/90.0,  -1.0/45.0,   2.0/45.0},
    { 32.0/45.0,  16.0/45.0,   8.0/45.0},
    { 32.0/45.0, -16.0/45.0,   8.0/45.0},
    {      0.00,       0.00,       1.00},
  };
};

constant constexpr const float WinogradTransforms<6, 3, 8>::wt_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::in_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::out_transform[8][8];

template <typename T,
          int BC = 32,
          int BO = 4,
          int M = 6,
          int R = 3>
[[kernel, max_total_threads_per_threadgroup(BO * 32)]] void winograd_conv_2d_weight_transform(
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
  simdgroup_matrix<T, 8, 8> G;
  G.thread_elements()[0] = WGT::wt_transform[sm][sn];
  G.thread_elements()[1] = WGT::wt_transform[sm][sn + 1];

  // Initialize Gt matrix
  simdgroup_matrix<T, 8, 8> Gt;
  Gt.thread_elements()[0] = WGT::wt_transform[sn][sm];
  Gt.thread_elements()[1] = WGT::wt_transform[sn + 1][sm];

  // Move to the correct output filter
  size_t ko = BO * tid + simd_group_id;
  wt_in  += ko * R * R * C;

  // wt_out is stored transposed (A x A x C x O)
  short ohw_0 = sm * 8 + sn;
  short ohw_1 = sm * 8 + sn + 1;
  device T* wt_out_0 = wt_out + ohw_0 * C * O + ko;
  device T* wt_out_1 = wt_out + ohw_1 * C * O + ko; 

  // Prepare shared memory
  threadgroup T Ws[BO][R][R][BC];

  // Loop over C
  for(int bc = 0; bc < C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for(int kh = 0; kh < R; ++kh) {
      for(int kw = 0; kw < R; ++kw) {
        for(int kc = simd_lane_id; kc < BC; kc += 32) {
          Ws[simd_group_id][kh][kw][kc] = wt_in[kh * R * C + kw * C + kc];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result 
    for(int c = 0; c < BC; ++c) {
      simdgroup_matrix<T, 8, 8> g;
      g.thread_elements()[0] = sm < R && sn < R ? Ws[simd_group_id][sm][sn][c] : T(0);
      g.thread_elements()[1] = sm < R && sn + 1 < R ? Ws[simd_group_id][sm][sn + 1][c] : T(0);

      simdgroup_matrix<T, 8, 8> g_out = (G * g) * Gt;
      wt_out_0[c * O] = g_out.thread_elements()[0];
      wt_out_1[c * O] = g_out.thread_elements()[1];
    }

    wt_in += BC;
    wt_out_0 += BC * O;
    wt_out_1 += BC * O;
  }

}

#define instantiate_winograd_conv_2d_weight_transform_base(name, itype, bc) \
  template [[host_name("winograd_conv_2d_weight_transform_" #name "_bc" #bc)]]\
  [[kernel]] void winograd_conv_2d_weight_transform<itype, bc>(\
      const device itype* wt_in [[buffer(0)]],\
      device itype* wt_out [[buffer(1)]],\
      const constant int& C [[buffer(2)]],\
      const constant int& O [[buffer(3)]],\
      uint tid [[threadgroup_position_in_grid]],\
      uint simd_group_id [[simdgroup_index_in_threadgroup]],\
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T,
          int BC,
          int WM,
          int WN,
          int M = 6,
          int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void winograd_conv_2d_input_transform(
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
  simdgroup_matrix<T, 8, 8> B;
  B.thread_elements()[0] = WGT::in_transform[sm][sn];
  B.thread_elements()[1] = WGT::in_transform[sm][sn + 1];

  // Initialize Bt matrix
  simdgroup_matrix<T, 8, 8> Bt;
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
  inp_in += tid.z * params.in_strides[0] 
             + bh * params.in_strides[1]
             + bw * params.in_strides[2];

  // Pre compute strides 
  int jump_in[TH][TW];

  for(int h = 0; h < TH; h++) {
    for(int w = 0; w < TW; w++) {
       jump_in[h][w] = h * params.in_strides[1] + w * params.in_strides[2];
    }
  }

  // inp_out is stored interleaved (A x A x tiles x C)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id = tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  device T* inp_out_0 = inp_out + ohw_0 * N_TILES * params.C + tile_id * params.C;
  device T* inp_out_1 = inp_out + ohw_1 * N_TILES * params.C + tile_id * params.C;

  // Prepare shared memory
  threadgroup T Is[A][A][BC];

  // Loop over C
  for(int bc = 0; bc < params.C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for(int h = 0; h < TH; h++) {
      for(int w = 0; w < TW; w++) {
        const device T* in_ptr = inp_in + jump_in[h][w];
        for(int c = simd_lane_id; c < BC; c += 32) {
          Is[kh + h][kw + w][c] = in_ptr[c];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result 
    for(int c = simd_group_id; c < BC; c += N_SIMD_GROUPS) {
      simdgroup_matrix<T, 8, 8> I;
      I.thread_elements()[0] = Is[sm][sn][c];
      I.thread_elements()[1] = Is[sm][sn + 1][c];

      simdgroup_matrix<T, 8, 8> I_out = (Bt * I) * B;
      inp_out_0[c] = I_out.thread_elements()[0];
      inp_out_1[c] = I_out.thread_elements()[1];
    }

    inp_in += BC;
    inp_out_0 += BC;
    inp_out_1 += BC;
  }

}

#define instantiate_winograd_conv_2d_input_transform(name, itype, bc) \
  template [[host_name("winograd_conv_2d_input_transform_" #name "_bc" #bc)]]\
  [[kernel]] void winograd_conv_2d_input_transform<itype, bc, 2, 2>(\
      const device itype* inp_in [[buffer(0)]],\
      device itype* inp_out [[buffer(1)]],\
      const constant MLXConvParams<2>& params [[buffer(2)]],\
      uint3 tid [[threadgroup_position_in_grid]],\
      uint3 lid [[thread_position_in_threadgroup]],\
      uint3 tgp_per_grid [[threadgroups_per_grid]],\
      uint simd_group_id [[simdgroup_index_in_threadgroup]],\
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T,
          int BO,
          int WM,
          int WN,
          int M = 6,
          int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void winograd_conv_2d_output_transform(
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
  simdgroup_matrix<T, 8, 8> B;
  B.thread_elements()[0] = WGT::out_transform[sm][sn];
  B.thread_elements()[1] = WGT::out_transform[sm][sn + 1];

  // Initialize At matrix
  simdgroup_matrix<T, 8, 8> Bt;
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
  out_out += tid.z * params.out_strides[0] 
              + bh * params.out_strides[1]
              + bw * params.out_strides[2];

  // Pre compute strides 
  int jump_in[TH][TW];

  for(int h = 0; h < TH; h++) {
    for(int w = 0; w < TW; w++) {
      bool valid = ((bh + h) < params.oS[0]) && ((bw + w) < params.oS[1]);
      jump_in[h][w] = valid ? h * params.out_strides[1] + w * params.out_strides[2] : -1;
    }
  }

  // out_in is stored interleaved (A x A x tiles x O)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id = tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  const device T* out_in_0 = out_in + ohw_0 * N_TILES * params.O + tile_id * params.O;
  const device T* out_in_1 = out_in + ohw_1 * N_TILES * params.O + tile_id * params.O;

  // Prepare shared memory
  threadgroup T Os[M][M][BO];

  // Loop over O
  for(int bo = 0; bo < params.O; bo += BO) {

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result 
    for(int c = simd_group_id; c < BO; c += N_SIMD_GROUPS) {
      simdgroup_matrix<T, 8, 8> O_mat;
      O_mat.thread_elements()[0] = out_in_0[c];
      O_mat.thread_elements()[1] = out_in_1[c];

      simdgroup_matrix<T, 8, 8> O_out = (Bt * (O_mat * B));
      if((sm < M) && (sn < M)) {
        Os[sm][sn][c] = O_out.thread_elements()[0];
      }
      if((sm < M) && ((sn + 1) < M)) {
        Os[sm][sn + 1][c] = O_out.thread_elements()[1];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read out from shared memory
    for(int h = 0; h < TH; h++) {
      for(int w = 0; w < TW; w++) {
        if(jump_in[h][w] >= 0) {
          device T* out_ptr = out_out + jump_in[h][w];
          for(int c = simd_lane_id; c < BO; c += 32) {
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
  template [[host_name("winograd_conv_2d_output_transform_" #name "_bo" #bo)]]\
  [[kernel]] void winograd_conv_2d_output_transform<itype, bo, 2, 2>(\
      const device itype* out_in [[buffer(0)]],\
      device itype* out_out [[buffer(1)]],\
      const constant MLXConvParams<2>& params [[buffer(2)]],\
      uint3 tid [[threadgroup_position_in_grid]],\
      uint3 lid [[thread_position_in_threadgroup]],\
      uint3 tgp_per_grid [[threadgroups_per_grid]],\
      uint simd_group_id [[simdgroup_index_in_threadgroup]],\
      uint simd_lane_id [[thread_index_in_simdgroup]]);

#define instantiate_winograd_conv_2d(name, itype) \
  instantiate_winograd_conv_2d_weight_transform_base(name, itype, 32) \
  instantiate_winograd_conv_2d_input_transform(name, itype, 32) \
  instantiate_winograd_conv_2d_output_transform(name, itype, 32)

instantiate_winograd_conv_2d(float32, float);
instantiate_winograd_conv_2d(float16, half);