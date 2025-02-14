// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

using namespace mlx::steel;

namespace mlx::core {

namespace {

template <int N>
void explicit_gemm_conv_ND_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<N>& conv_params) {
  // Get gemm shapes
  int implicit_M = out.size() / conv_params.O;
  int implicit_K = wt.size() / conv_params.O;
  int implicit_N = conv_params.O;
  // Prepare unfolding array
  Shape unfolded_shape{implicit_M, implicit_K};
  array in_unfolded(unfolded_shape, in.dtype(), nullptr, {});

  in_unfolded.set_data(allocator::malloc_or_wait(in_unfolded.nbytes()));

  // Prepare unfolding kernel
  std::ostringstream kname;
  kname << "naive_unfold_nd_" << type_to_name(in_unfolded) << "_" << N;
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(in_unfolded, 1);

  compute_encoder.set_bytes(conv_params, 2);

  // Launch unfolding kernel
  size_t tgp_x = std::min(conv_params.C, 64);
  tgp_x = 32 * ((tgp_x + 32 - 1) / 32);
  size_t tgp_y = 256 / tgp_x;

  MTL::Size grid_dims = MTL::Size(
      conv_params.C, unfolded_shape[1] / conv_params.C, unfolded_shape[0]);
  MTL::Size group_dims = MTL::Size(
      std::min(tgp_x, grid_dims.width), std::min(tgp_y, grid_dims.height), 1);

  compute_encoder.dispatch_threads(grid_dims, group_dims);

  // Reshape weight
  Shape wt_reshape{implicit_K, implicit_N};
  Strides wt_restride{1, implicit_K};
  array wt_reshaped(wt_reshape, wt.dtype(), nullptr, {});
  auto wt_flags = wt.flags();
  wt_flags.row_contiguous = false;
  wt_flags.col_contiguous = true;
  wt_reshaped.copy_shared_buffer(wt, wt_restride, wt_flags, wt.data_size());

  // Perform gemm
  std::vector<array> copies = {in_unfolded};
  return steel_matmul(
      s,
      d,
      /*a = */ in_unfolded,
      /*b = */ wt_reshaped,
      /*c = */ out,
      /*M = */ implicit_M,
      /*N = */ implicit_N,
      /*K = */ implicit_K,
      /*batch_size_out = */ 1,
      /*a_cols = */ implicit_K,
      /*b_cols = */ implicit_K,
      /*a_transposed = */ false,
      /*b_transposed = */ true,
      /*copies = */ copies);
}

template <int N>
void explicit_gemm_conv_group_ND_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<N>& conv_params) {
  const int groups = conv_params.groups;
  const int C_per_group = conv_params.C / conv_params.groups;
  const int O_per_group = conv_params.O / conv_params.groups;
  // Get gemm shapes
  const int implicit_M = out.size() / conv_params.O;
  const int implicit_K = wt.size() / conv_params.O;
  const int implicit_N = O_per_group;

  int kernel_size = 1;
  for (int i = 0; i < N; ++i) {
    kernel_size *= conv_params.wS[i];
  }

  // Prepare unfolding array
  Shape unfolded_shape{implicit_M, implicit_K * groups};
  array in_unfolded(unfolded_shape, in.dtype(), nullptr, {});
  in_unfolded.set_data(allocator::malloc_or_wait(in_unfolded.nbytes()));

  // Prepare unfolding kernel
  std::ostringstream kname;
  kname << "naive_unfold_transpose_nd_" << type_to_name(in_unfolded) << "_"
        << N;
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(in_unfolded, 1);

  compute_encoder.set_bytes(conv_params, 2);

  // Launch unfolding kernel
  size_t tgp_x = std::min(conv_params.C, 64);
  tgp_x = 32 * ((tgp_x + 32 - 1) / 32);
  size_t tgp_y = 256 / tgp_x;

  MTL::Size grid_dims = MTL::Size(
      conv_params.C, unfolded_shape[1] / conv_params.C, unfolded_shape[0]);
  MTL::Size group_dims = MTL::Size(
      std::min(tgp_x, grid_dims.width), std::min(tgp_y, grid_dims.height), 1);

  compute_encoder.dispatch_threads(grid_dims, group_dims);

  // Transpose kernel weights so that we can slice them by contiguous chunks
  // of channel groups.
  array wt_view(
      {wt.shape(0), C_per_group, kernel_size}, wt.dtype(), nullptr, {});
  wt_view.copy_shared_buffer(
      wt, {wt.strides(0), 1, C_per_group}, wt.flags(), wt.size());

  // Materialize
  auto wt_transpose = array(wt_view.shape(), wt_view.dtype(), nullptr, {});
  copy_gpu(wt_view, wt_transpose, CopyType::General, s);

  // Perform gemm
  std::vector<array> copies = {in_unfolded, wt_transpose};
  return steel_matmul_regular(
      s,
      d,
      /* a = */ in_unfolded,
      /* b = */ wt_transpose,
      /* c = */ out,
      /* M = */ implicit_M,
      /* N = */ implicit_N,
      /* K = */ implicit_K,
      /* batch_size_out = */ groups,
      /* a_cols = */ implicit_K * groups,
      /* b_cols = */ implicit_K,
      /* out_cols = */ implicit_N * groups,
      /* a_transposed = */ false,
      /* b_transposed = */ true,
      /* batch_shape = */ {1},
      /* batch_strides = */ {0},
      /* A_batch_strides = */ size_t(implicit_K),
      /* B_batch_strides = */ size_t(implicit_N) * implicit_K,
      /* matrix_stride_out = */ size_t(implicit_N),
      /*copies = */ copies);
}

void conv_1D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    int groups,
    bool flip) {
  // Make conv params
  MLXConvParams<1> conv_params{
      /* const int  N = */ static_cast<int>(in.shape(0)),
      /* const int  C = */ static_cast<int>(in.shape(2)),
      /* const int  O = */ static_cast<int>(wt.shape(0)),
      /* const int iS[NDIM] = */ {static_cast<int>(in.shape(1))},
      /* const int wS[NDIM] = */ {static_cast<int>(wt.shape(1))},
      /* const int oS[NDIM] = */ {static_cast<int>(out.shape(1))},
      /* const int str[NDIM] = */ {wt_strides[0]},
      /* const int pad[NDIM] = */ {padding[0]},
      /* const int kdil[NDIM] = */ {wt_dilation[0]},
      /* const int idil[NDIM] = */ {in_dilation[0]},
      /* const size_t in_strides[NDIM + 2] = */
      {in.strides()[0], in.strides()[1], in.strides()[2]},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt.strides()[0], wt.strides()[1], wt.strides()[2]},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides()[0], out.strides()[1], out.strides()[2]},
      /* const int groups = */ groups,
      /* const bool flip = */ flip};

  // Direct to explicit gemm conv
  if (groups > 1) {
    return explicit_gemm_conv_group_ND_gpu(s, d, in, wt, out, conv_params);
  } else {
    return explicit_gemm_conv_ND_gpu(s, d, in, wt, out, conv_params);
  }
}

void slow_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params) {
  int bm = 16, bn = 8;
  int tm = 4, tn = 4;

  std::ostringstream kname;
  kname << "naive_conv_2d_" << type_to_name(out) << "_bm" << bm << "_bn" << bn
        << "_tm" << tm << "_tn" << tn;

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder.set_compute_pipeline_state(kernel);

  size_t n_pixels = conv_params.oS[0] * conv_params.oS[1];

  size_t grid_dim_x = (n_pixels + (tm * bm) - 1) / (tm * bm);
  size_t grid_dim_y = (conv_params.O + (tn * bn) - 1) / (tn * bn);
  size_t grid_dim_z = conv_params.N;

  MTL::Size group_dims = MTL::Size(bm, bn, 1);
  MTL::Size grid_dims = MTL::Size(grid_dim_x, grid_dim_y, grid_dim_z);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(wt, 1);
  compute_encoder.set_output_array(out, 2);

  compute_encoder.set_bytes(conv_params, 3);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void implicit_gemm_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params) {
  const int groups = conv_params.groups;
  const int C_per_group = conv_params.C / conv_params.groups;
  const int O_per_group = conv_params.O / conv_params.groups;

  // Deduce implicit gemm size
  const int implicit_M = conv_params.N * conv_params.oS[0] * conv_params.oS[1];
  const int implicit_N = O_per_group;
  const int implicit_K = conv_params.wS[0] * conv_params.wS[1] * C_per_group;

  // Determine block and warp tiles
  int wm = 2, wn = 2;

  int bm = implicit_M >= 8192 && C_per_group >= 64 ? 64 : 32;
  int bn = (bm == 64 || implicit_N >= 64) ? 64 : 32;
  int bk = 16;

  if (implicit_N <= 16) {
    bn = 8;
    wm = 4;
    wn = 1;
  }

  int tn = (implicit_N + bn - 1) / bn;
  int tm = (implicit_M + bm - 1) / bm;
  int swizzle_log = 0;

  // Fix small channel specialization
  int n_channel_specialization = 0;
  int channel_k_iters = ((C_per_group + bk - 1) / bk);
  int gemm_k_iters = conv_params.wS[0] * conv_params.wS[1] * channel_k_iters;

  if (C_per_group <= 2) {
    gemm_k_iters = (implicit_K + bk - 1) / bk;
    n_channel_specialization = C_per_group;
  } else if (C_per_group <= 4) {
    gemm_k_iters = ((conv_params.wS[0] * conv_params.wS[1] * 4) + bk - 1) / bk;
    n_channel_specialization = C_per_group;
  }

  bool small_filter = (!n_channel_specialization) &&
      (conv_params.wS[0] <= 16 && conv_params.wS[1] <= 16);

  // Fix host side helper params
  int sign = (conv_params.flip ? -1 : 1);
  int ijw = conv_params.in_strides[2] * conv_params.kdil[1];
  int ijh = conv_params.in_strides[1] * conv_params.kdil[0];

  int inp_jump_w = sign * ijw;
  int inp_jump_h = sign * (ijh - (conv_params.wS[1] - 1) * ijw);
  int inp_jump_c = bk - sign * (conv_params.wS[0] - 1) * ijh -
      sign * (conv_params.wS[1] - 1) * ijw;

  // Build implicit gemm params
  ImplicitGemmConv2DParams gemm_params{
      /* const int M = */ implicit_M,
      /* const int N = */ implicit_N,
      /* const int K = */ implicit_K,

      /* const int gemm_k_iterations = */ gemm_k_iters,

      /* const int inp_jump_w = */ inp_jump_w,
      /* const int inp_jump_h = */ inp_jump_h,
      /* const int inp_jump_c = */ inp_jump_c,

      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const int swizzle_log = */ swizzle_log};

  // Determine kernel
  std::ostringstream kname;
  kname << "implicit_gemm_conv_2d_" << type_to_name(out) << "_bm" << bm << "_bn"
        << bn << "_bk" << bk << "_wm" << wm << "_wn" << wn << "_channel_"
        << (n_channel_specialization ? std::to_string(n_channel_specialization)
                                     : "l")
        << "_filter_" << (small_filter ? 's' : 'l');

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_conv_kernel(
      d,
      kname.str(),
      out,
      bm,
      bn,
      bk,
      wm,
      wn,
      n_channel_specialization,
      small_filter);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Deduce grid launch dimensions
  int tile = 1 << swizzle_log;
  size_t grid_dim_y = (tm + tile - 1) / tile;
  size_t grid_dim_x = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(grid_dim_x, grid_dim_y, groups);

  // Encode arrays
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(wt, 1);
  compute_encoder.set_output_array(out, 2);

  // Encode params
  compute_encoder.set_bytes(conv_params, 3);
  compute_encoder.set_bytes(gemm_params, 4);

  // Launch kernel
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void implicit_gemm_conv_2D_general_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params) {
  // Deduce implicit gemm size
  int implicit_M = conv_params.N * conv_params.oS[0] * conv_params.oS[1];
  int implicit_N = conv_params.O;
  int implicit_K = conv_params.wS[0] * conv_params.wS[1] * conv_params.C;

  // Determine block and warp tiles
  int wm = 2, wn = 2;

  // Make jump params
  int f_wgt_jump_h =
      std::lcm(conv_params.idil[0], conv_params.kdil[0]) / conv_params.kdil[0];
  int f_wgt_jump_w =
      std::lcm(conv_params.idil[1], conv_params.kdil[1]) / conv_params.kdil[1];

  int f_out_jump_h =
      std::lcm(conv_params.idil[0], conv_params.str[0]) / conv_params.str[0];
  int f_out_jump_w =
      std::lcm(conv_params.idil[1], conv_params.str[1]) / conv_params.str[1];

  int adj_out_h = (conv_params.oS[0] + f_out_jump_h - 1) / f_out_jump_h;
  int adj_out_w = (conv_params.oS[1] + f_out_jump_w - 1) / f_out_jump_w;
  int adj_out_hw = adj_out_h * adj_out_w;
  int adj_implicit_m = conv_params.N * adj_out_hw;

  Conv2DGeneralJumpParams jump_params{
      /* const int f_wgt_jump_h = */ f_wgt_jump_h,
      /* const int f_wgt_jump_w = */ f_wgt_jump_w,

      /* const int f_out_jump_h = */ f_out_jump_h,
      /* const int f_out_jump_w = */ f_out_jump_w,

      /* const int adj_out_h = */ adj_out_h,
      /* const int adj_out_w = */ adj_out_w,
      /* const int adj_out_hw = */ adj_out_hw,
      /* const int adj_implicit_m = */ adj_implicit_m};

  // Make base info
  std::vector<Conv2DGeneralBaseInfo> base_h(f_out_jump_h);
  std::vector<Conv2DGeneralBaseInfo> base_w(f_out_jump_w);

  int jump_h = conv_params.flip ? -conv_params.kdil[0] : conv_params.kdil[0];
  int jump_w = conv_params.flip ? -conv_params.kdil[1] : conv_params.kdil[1];

  int init_h =
      (conv_params.flip ? (conv_params.wS[0] - 1) * conv_params.kdil[0] : 0);
  int init_w =
      (conv_params.flip ? (conv_params.wS[1] - 1) * conv_params.kdil[1] : 0);

  for (int i = 0; i < f_out_jump_h; ++i) {
    int ih_loop = i * conv_params.str[0] - conv_params.pad[0] + init_h;

    int wh_base = 0;
    while (wh_base < conv_params.wS[0] && ih_loop % conv_params.idil[0] != 0) {
      wh_base++;
      ih_loop += jump_h;
    }

    int wh_size =
        ((conv_params.wS[0] - wh_base) + f_wgt_jump_h - 1) / f_wgt_jump_h;
    base_h[i] = {wh_base, wh_size};
  }

  for (int j = 0; j < f_out_jump_w; ++j) {
    int iw_loop = j * conv_params.str[1] - conv_params.pad[1] + init_w;

    int ww_base = 0;
    while (ww_base < conv_params.wS[1] && iw_loop % conv_params.idil[1] != 0) {
      ww_base++;
      iw_loop += jump_w;
    }

    int ww_size =
        ((conv_params.wS[1] - ww_base) + f_wgt_jump_w - 1) / f_wgt_jump_w;
    base_w[j] = {ww_base, ww_size};
  }

  // Collect block sizes
  int bm = adj_implicit_m >= 8192 && conv_params.C >= 64 ? 64 : 32;
  int bn = (bm == 64 && implicit_N >= 64) ? 64 : 32;
  int bk = 16;

  int tn = (implicit_N + bn - 1) / bn;
  int tm = (adj_implicit_m + bm - 1) / bm;
  int swizzle_log = 0;

  // Get channel iteration info
  int channel_k_iters = ((conv_params.C + bk - 1) / bk);
  int gemm_k_iters = channel_k_iters;

  // Fix host side helper params
  int sign = (conv_params.flip ? -1 : 1);
  int ijw = conv_params.in_strides[2] * conv_params.kdil[1];
  int ijh = conv_params.in_strides[1] * conv_params.kdil[0];

  int inp_jump_w = sign * ijw;
  int inp_jump_h = sign * (ijh - (conv_params.wS[1] - 1) * ijw);
  int inp_jump_c = bk - sign * (conv_params.wS[0] - 1) * ijh -
      sign * (conv_params.wS[1] - 1) * ijw;

  // Build implicit gemm params
  ImplicitGemmConv2DParams gemm_params{
      /* const int M = */ implicit_M,
      /* const int N = */ implicit_N,
      /* const int K = */ implicit_K,

      /* const int gemm_k_iterations = */ gemm_k_iters,

      /* const int inp_jump_w = */ inp_jump_w,
      /* const int inp_jump_h = */ inp_jump_h,
      /* const int inp_jump_c = */ inp_jump_c,

      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const int swizzle_log = */ swizzle_log};

  // Determine kernel
  std::ostringstream kname;
  kname << "implicit_gemm_conv_2d_general_" << type_to_name(out) << "_bm" << bm
        << "_bn" << bn << "_bk" << bk << "_wm" << wm << "_wn" << wn;

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel =
      get_steel_conv_general_kernel(d, kname.str(), out, bm, bn, bk, wm, wn);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Deduce grid launch dimensions
  int tile = 1 << swizzle_log;
  size_t grid_dim_y = (tm + tile - 1) / tile;
  size_t grid_dim_x = tn * tile;
  size_t grid_dim_z = f_out_jump_h * f_out_jump_w;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(grid_dim_x, grid_dim_y, grid_dim_z);

  // Encode arrays
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(wt, 1);
  compute_encoder.set_output_array(out, 2);

  // Encode params
  compute_encoder.set_bytes(conv_params, 3);
  compute_encoder.set_bytes(gemm_params, 4);
  compute_encoder.set_bytes(jump_params, 5);

  compute_encoder.set_vector_bytes(base_h, 6);
  compute_encoder.set_vector_bytes(base_w, 7);

  // Launch kernel
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void winograd_conv_2D_fused_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params,
    std::vector<array>& copies_w) {
  int O_c = conv_params.O;
  int C_c = conv_params.C;

  int N_tiles_n = conv_params.N;
  int N_tiles_h = (conv_params.oS[0] + 1) / 2;
  int N_tiles_w = (conv_params.oS[1] + 1) / 2;
  int N_tiles = N_tiles_n * N_tiles_h * N_tiles_w;

  int bc = 32;
  int wm = 4;
  int wn = 1;
  std::ostringstream kname;
  kname << "winograd_conv_2d_fused_" << type_to_name(out) << "_flip"
        << conv_params.flip;
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(wt, 1);
  compute_encoder.set_output_array(out, 2);

  compute_encoder.set_bytes(conv_params, 3);

  MTL::Size group_dims = MTL::Size(8, 8, 2);
  MTL::Size grid_dims =
      MTL::Size(O_c / 8, (N_tiles_h * N_tiles_w) / 8, N_tiles_n);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void winograd_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params,
    std::vector<array>& copies_w) {
  int O_c = conv_params.O;
  int C_c = conv_params.C;

  int N_tiles_n = conv_params.N;
  int N_tiles_h = (conv_params.oS[0] + 5) / 6;
  int N_tiles_w = (conv_params.oS[1] + 5) / 6;
  int N_tiles = N_tiles_n * N_tiles_h * N_tiles_w;

  // Do filter transform
  Shape filt_wg_shape = {8 * 8, conv_params.C, conv_params.O};
  array filt_wg(std::move(filt_wg_shape), wt.dtype(), nullptr, {});
  filt_wg.set_data(allocator::malloc_or_wait(filt_wg.nbytes()));
  copies_w.push_back(filt_wg);
  {
    int bc = 32;
    int bo = 4;
    std::ostringstream kname;
    kname << "winograd_conv_2d_weight_transform_" << type_to_name(out) << "_bc"
          << bc << "_flip" << conv_params.flip;
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(wt, 0);
    compute_encoder.set_output_array(filt_wg, 1);

    compute_encoder.set_bytes(C_c, 2);
    compute_encoder.set_bytes(O_c, 3);

    MTL::Size group_dims = MTL::Size(32, bo, 1);
    MTL::Size grid_dims = MTL::Size(O_c / bo, 1, 1);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Do input transform
  Shape inp_wg_shape = {8 * 8, N_tiles, conv_params.C};
  array inp_wg(std::move(inp_wg_shape), in.dtype(), nullptr, {});
  inp_wg.set_data(allocator::malloc_or_wait(inp_wg.nbytes()));
  copies_w.push_back(inp_wg);
  {
    int bc = 32;
    int wm = 2;
    int wn = 2;
    std::ostringstream kname;
    kname << "winograd_conv_2d_input_transform_" << type_to_name(out) << "_bc"
          << bc;
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(inp_wg, 1);

    compute_encoder.set_bytes(conv_params, 2);

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(N_tiles_w, N_tiles_h, N_tiles_n);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Do batched gemm
  Shape out_wg_shape = {8 * 8, N_tiles, conv_params.O};
  array out_wg(std::move(out_wg_shape), in.dtype(), nullptr, {});
  out_wg.set_data(allocator::malloc_or_wait(out_wg.nbytes()));
  copies_w.push_back(out_wg);
  {
    std::vector<array> empty_copies;
    steel_matmul(
        s,
        d,
        /*a = */ inp_wg,
        /*b = */ filt_wg,
        /*c = */ out_wg,
        /*M = */ N_tiles,
        /*N = */ conv_params.O,
        /*K = */ conv_params.C,
        /*batch_size_out = */ 8 * 8,
        /*a_cols = */ conv_params.C,
        /*b_cols = */ conv_params.O,
        /*a_transposed = */ false,
        /*b_transposed = */ false,
        /*copies = */ empty_copies);
  }

  // Do output transform
  {
    int bc = 32;
    int wm = 2;
    int wn = 2;
    std::ostringstream kname;
    kname << "winograd_conv_2d_output_transform_" << type_to_name(out) << "_bo"
          << bc;
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(out_wg, 0);
    compute_encoder.set_output_array(out, 1);

    compute_encoder.set_bytes(conv_params, 2);

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(N_tiles_w, N_tiles_h, N_tiles_n);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    const int groups,
    bool flip,
    std::vector<array>& copies) {
  // Make conv params
  MLXConvParams<2> conv_params{
      /* const int  N = */ static_cast<int>(in.shape(0)),
      /* const int  C = */ static_cast<int>(in.shape(3)),
      /* const int  O = */ static_cast<int>(wt.shape(0)),
      /* const int iS[NDIM] = */
      {static_cast<int>(in.shape(1)), static_cast<int>(in.shape(2))},
      /* const int wS[NDIM] = */
      {static_cast<int>(wt.shape(1)), static_cast<int>(wt.shape(2))},
      /* const int oS[NDIM] = */
      {static_cast<int>(out.shape(1)), static_cast<int>(out.shape(2))},
      /* const int str[NDIM] = */ {wt_strides[0], wt_strides[1]},
      /* const int pad[NDIM] = */ {padding[0], padding[1]},
      /* const int kdil[NDIM] = */ {wt_dilation[0], wt_dilation[1]},
      /* const int idil[NDIM] = */ {in_dilation[0], in_dilation[1]},
      /* const size_t in_strides[NDIM + 2] = */
      {in.strides(0), in.strides(1), in.strides(2), in.strides(3)},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt.strides(0), wt.strides(1), wt.strides(2), wt.strides(3)},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides(0), out.strides(1), out.strides(2), out.strides(3)},
      /* const int groups = */ groups,
      /* const bool flip = */ flip,
  };

  bool is_stride_one = conv_params.str[0] == 1 && conv_params.str[1] == 1;
  bool is_kdil_one = conv_params.kdil[0] == 1 && conv_params.kdil[1] == 1;
  bool is_idil_one = conv_params.idil[0] == 1 && conv_params.idil[1] == 1;

  if (groups > 1) {
    const int C_per_group = conv_params.C / groups;
    const int O_per_group = conv_params.O / groups;

    if (is_idil_one && (C_per_group <= 4 || C_per_group % 16 == 0) &&
        (O_per_group <= 16 || O_per_group % 16 == 0)) {
      return implicit_gemm_conv_2D_gpu(s, d, in, wt, out, conv_params);
    } else {
      return explicit_gemm_conv_group_ND_gpu(s, d, in, wt, out, conv_params);
    }
  }

  // Direct to winograd conv
  bool img_large =
      (conv_params.N * conv_params.iS[0] * conv_params.iS[1]) >= 1ul << 12;
  bool channels_large = (conv_params.C + conv_params.O) >= 256;
  if (conv_params.wS[0] == 3 && conv_params.wS[1] == 3 &&
      conv_params.C % 32 == 0 && conv_params.O % 32 == 0 && is_stride_one &&
      is_kdil_one && is_idil_one) {
    if (img_large && channels_large) {
      return winograd_conv_2D_gpu(s, d, in, wt, out, conv_params, copies);
    }
    if (conv_params.N <= 1) {
      return winograd_conv_2D_fused_gpu(s, d, in, wt, out, conv_params, copies);
    }
  }

  // Direct to implicit gemm conv
  if (is_idil_one && (conv_params.C <= 4 || conv_params.C % 16 == 0) &&
      (conv_params.O <= 16 || conv_params.O % 16 == 0)) {
    return implicit_gemm_conv_2D_gpu(s, d, in, wt, out, conv_params);
  }

  else if (conv_params.C % 16 == 0 && conv_params.O % 16 == 0) {
    return implicit_gemm_conv_2D_general_gpu(s, d, in, wt, out, conv_params);
  }

  // Direct to explicit gemm conv
  else {
    return explicit_gemm_conv_ND_gpu(s, d, in, wt, out, conv_params);
  }
}

void conv_3D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    bool flip,
    std::vector<array>& copies) {
  // Make conv params
  MLXConvParams<3> conv_params{
      /* const int  N = */ static_cast<int>(in.shape(0)),
      /* const int  C = */ static_cast<int>(in.shape(4)),
      /* const int  O = */ static_cast<int>(wt.shape(0)),
      /* const int iS[NDIM] = */
      {static_cast<int>(in.shape(1)),
       static_cast<int>(in.shape(2)),
       static_cast<int>(in.shape(3))},
      /* const int wS[NDIM] = */
      {static_cast<int>(wt.shape(1)),
       static_cast<int>(wt.shape(2)),
       static_cast<int>(wt.shape(3))},
      /* const int oS[NDIM] = */
      {static_cast<int>(out.shape(1)),
       static_cast<int>(out.shape(2)),
       static_cast<int>(out.shape(3))},
      /* const int str[NDIM] = */ {wt_strides[0], wt_strides[1], wt_strides[2]},
      /* const int pad[NDIM] = */ {padding[0], padding[1], padding[2]},
      /* const int kdil[NDIM] = */
      {wt_dilation[0], wt_dilation[1], wt_dilation[2]},
      /* const int idil[NDIM] = */
      {in_dilation[0], in_dilation[1], in_dilation[2]},
      /* const size_t in_strides[NDIM + 2] = */
      {in.strides()[0],
       in.strides()[1],
       in.strides()[2],
       in.strides()[3],
       in.strides()[4]},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt.strides()[0],
       wt.strides()[1],
       wt.strides()[2],
       wt.strides()[3],
       wt.strides()[4]},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides()[0],
       out.strides()[1],
       out.strides()[2],
       out.strides()[3],
       out.strides()[4]},
      /* const int groups = */ 1,
      /* const bool flip = */ flip,
  };
  return explicit_gemm_conv_ND_gpu(s, d, in, wt, out, conv_params);
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure contiguity
  std::vector<array> copies;
  auto in = inputs[0];
  auto wt = inputs[1];
  if (!in.flags().row_contiguous) {
    array arr_copy(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, arr_copy, CopyType::General, s);
    copies.push_back(arr_copy);
    in = arr_copy;
  }
  if (!wt.flags().row_contiguous) {
    array arr_copy(wt.shape(), wt.dtype(), nullptr, {});
    copy_gpu(wt, arr_copy, CopyType::General, s);
    copies.push_back(arr_copy);
    wt = arr_copy;
  }

  // Check for 1x1 conv
  auto is_one = [](int x) { return x == 1; };
  auto is_zero = [](int x) { return x == 0; };
  if (groups_ == 1 && (wt.shape(0) * wt.shape(-1) == wt.size()) &&
      std::all_of(wt.shape().begin() + 1, wt.shape().end() - 1, is_one) &&
      std::all_of(kernel_strides_.begin(), kernel_strides_.end(), is_one) &&
      std::all_of(input_dilation_.begin(), input_dilation_.end(), is_one) &&
      std::all_of(kernel_dilation_.begin(), kernel_dilation_.end(), is_one) &&
      std::all_of(padding_.begin(), padding_.end(), is_zero)) {
    std::vector<array> empty_copies;
    steel_matmul_regular(
        s,
        d,
        /*a = */ in,
        /*b = */ wt,
        /*c = */ out,
        /*M = */ in.size() / in.shape(-1),
        /*N = */ wt.shape(0),
        /*K = */ in.shape(-1),
        /*batch_size_out = */ 1,
        /*lda = */ in.shape(-1),
        /*ldb = */ wt.shape(-1),
        /*ldd = */ wt.shape(0),
        /*transpose_a = */ false,
        /*transpose_b = */ true,
        /*batch_shape = */ {1},
        /*batch_strides = */ {1},
        /*A_batch_stride = */ 0,
        /*B_batch_stride = */ 0,
        /*matrix_stride_out = */ 0,
        /*copies = */ empty_copies);
  }
  // 3D conv
  else if (out.ndim() == 5) {
    conv_3D_gpu(
        s,
        d,
        in,
        wt,
        out,
        padding_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        flip_,
        copies);
  }
  // 2D conv
  else if (out.ndim() == 4) {
    conv_2D_gpu(
        s,
        d,
        in,
        wt,
        out,
        padding_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        groups_,
        flip_,
        copies);
  }
  // 1D conv
  else if (out.ndim() == 3) {
    conv_1D_gpu(
        s,
        d,
        in,
        wt,
        out,
        padding_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        groups_,
        flip_);
  }
  // Throw error
  else {
    throw std::invalid_argument(
        "[Convolution::eval_gpu] Only supports 1D, 2D or 3D convolutions.");
  }

  // Record copies
  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core
