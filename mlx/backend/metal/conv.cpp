// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cassert>
#include <numeric>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
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

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  auto result = contiguous_copy_gpu(x, s);
  d.add_temporary(result, s.index);
  return result;
}

template <int N>
void explicit_gemm_conv_ND_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<N>& conv_params) {
  // Get gemm shapes
  int implicit_M = out.size() / conv_params.O;
  int implicit_K = wt.size() / conv_params.O;
  int implicit_N = conv_params.O;
  // Prepare unfolding array
  Shape unfolded_shape{implicit_M, implicit_K};
  array in_unfolded(unfolded_shape, in.dtype(), nullptr, {});

  in_unfolded.set_data(allocator::malloc(in_unfolded.nbytes()));

  // Prepare unfolding kernel
  std::string kname;
  kname.reserve(32);
  concatenate(kname, "naive_unfold_nd_", type_to_name(in_unfolded), "_", N);
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
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
    array& out,
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
  in_unfolded.set_data(allocator::malloc(in_unfolded.nbytes()));

  // Prepare unfolding kernel
  std::string kname;
  kname.reserve(32);
  concatenate(
      kname, "naive_unfold_transpose_nd_", type_to_name(in_unfolded), "_", N);
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
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
  array wt_transpose = contiguous_copy_gpu(wt_view, s);

  // Perform gemm
  std::vector<array> copies = {in_unfolded, wt_transpose};
  return steel_matmul_regular(
      /* const Stream& s = */ s,
      /* Device& d = */ d,
      /* const array& a = */ in_unfolded,
      /* const array& b = */ wt_transpose,
      /* array& c = */ out,
      /* int M = */ implicit_M,
      /* int N = */ implicit_N,
      /* int K = */ implicit_K,
      /* int batch_size_out = */ groups,
      /* int lda = */ implicit_K * groups,
      /* int ldb = */ implicit_K,
      /* int ldd = */ implicit_N * groups,
      /* bool transpose_a = */ false,
      /* bool transpose_b = */ true,
      /* std::vector<array>& copies = */ copies,
      /* Shape batch_shape = */ {1},
      /* Strides batch_strides = */ {0},
      /* int64_t A_batch_strides = */ int64_t(implicit_K),
      /* int64_t B_batch_strides = */ int64_t(implicit_N) * implicit_K,
      /* int64_t matrix_stride_out = */ int64_t(implicit_N));
}

void implicit_gemm_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array& out,
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
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "implicit_gemm_conv_2d_",
      type_to_name(out),
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      "_channel_",
      n_channel_specialization ? std::to_string(n_channel_specialization) : "l",
      "_filter_",
      small_filter ? 's' : 'l');

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_conv_kernel(
      d,
      kname,
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
    array& out,
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
  bool align_C = conv_params.C % bk == 0;

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
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "implicit_gemm_conv_2d_general_",
      type_to_name(out),
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn);
  std::string hash_name;
  hash_name.reserve(64);
  concatenate(hash_name, kname, "_alC_", align_C);
  metal::MTLFCList func_consts = {
      {&align_C, MTL::DataType::DataTypeBool, 200},
  };

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_conv_general_kernel(
      d, kname, hash_name, func_consts, out, bm, bn, bk, wm, wn);
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

void implicit_gemm_conv_3D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<3>& conv_params) {
  const int groups = conv_params.groups;
  const int C_per_group = conv_params.C / conv_params.groups;
  const int O_per_group = conv_params.O / conv_params.groups;

  // Deduce implicit gemm size
  const int implicit_M =
      conv_params.N * conv_params.oS[0] * conv_params.oS[1] * conv_params.oS[2];
  const int implicit_N = O_per_group;
  const int implicit_K =
      conv_params.wS[0] * conv_params.wS[1] * conv_params.wS[2] * C_per_group;

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

  bool small_filter =
      (conv_params.wS[0] <= 16 && conv_params.wS[1] <= 16 &&
       conv_params.wS[2] <= 16);

  int channel_k_iters = ((C_per_group + bk - 1) / bk);
  int gemm_k_iters = conv_params.wS[0] * conv_params.wS[1] * conv_params.wS[2] *
      channel_k_iters;

  // Fix host side helper params
  int sign = (conv_params.flip ? -1 : 1);
  int ijw = conv_params.in_strides[3] * conv_params.kdil[2];
  int ijh = conv_params.in_strides[2] * conv_params.kdil[1];
  int ijd = conv_params.in_strides[1] * conv_params.kdil[0];

  int inp_jump_w = sign * ijw;
  int inp_jump_h = sign * (ijh - (conv_params.wS[2] - 1) * ijw);
  int inp_jump_d = sign *
      (ijd - (conv_params.wS[1] - 1) * ijh - (conv_params.wS[2] - 1) * ijw);
  int inp_jump_c = bk - sign * (conv_params.wS[0] - 1) * ijd -
      sign * (conv_params.wS[1] - 1) * ijh -
      sign * (conv_params.wS[2] - 1) * ijw;

  // Build implicit gemm params
  ImplicitGemmConv3DParams gemm_params{
      /* const int M = */ implicit_M,
      /* const int N = */ implicit_N,
      /* const int K = */ implicit_K,

      /* const int gemm_k_iterations = */ gemm_k_iters,

      /* const int inp_jump_w = */ inp_jump_w,
      /* const int inp_jump_h = */ inp_jump_h,
      /* const int inp_jump_d = */ inp_jump_d,
      /* const int inp_jump_c = */ inp_jump_c,

      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const int swizzle_log = */ swizzle_log};

  // Determine kernel
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "implicit_gemm_conv_3d_",
      type_to_name(out),
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      "_filter_",
      small_filter ? 's' : 'l');

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel =
      get_steel_conv_3d_kernel(d, kname, out, bm, bn, bk, wm, wn, small_filter);
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

void pad_and_slice_conv_3D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const MLXConvParams<3>& conv_params) {
  // For now assume conv_params.groups == 1
  int extra_c = ((conv_params.C + 15) / 16) * 16 - conv_params.C;
  int extra_o = ((conv_params.O + 15) / 16) * 16 - conv_params.O;

  // Pad function
  auto pad_array = [&](const array& x, int pad_ax_first, int pad_ax_last) {
    if (pad_ax_first == 0 && pad_ax_last == 0) {
      return ensure_row_contiguous(x, d, s);
    }

    auto xshape = x.shape();
    xshape.front() += pad_ax_first;
    xshape.back() += pad_ax_last;
    array x_copy(xshape, x.dtype(), nullptr, {});
    array zero(0, x.dtype());
    pad_gpu(x, zero, x_copy, {0, -1}, {0, 0}, s);
    d.add_temporary(x_copy, s.index);

    return x_copy;
  };

  // Allocate space for the intermediate output. Don't save it as a temporary
  // since it will be sliced to the output so they share the buffer.
  auto oshape = out.shape();
  oshape.back() += extra_o;
  array intermediate(oshape, out.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));

  // Actually pad and conv
  array in = pad_array(in_pre, 0, extra_c);
  array wt = pad_array(wt_pre, extra_o, extra_c);
  auto new_params =
      MLXConvParams<3>::with_padded_channels(conv_params, extra_o, extra_c);
  implicit_gemm_conv_3D_gpu(s, d, in, wt, intermediate, new_params);

  // Slice out
  out.copy_shared_buffer(
      intermediate, intermediate.strides(), {0}, intermediate.data_size());
}

void dispatch_conv_3D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const MLXConvParams<3>& conv_params,
    std::vector<array>& copies) {
  bool is_idil_one = conv_params.idil[0] == 1 && conv_params.idil[1] == 1 &&
      conv_params.idil[2] == 1;
  const int C_per_group = conv_params.C / conv_params.groups;
  const int O_per_group = conv_params.O / conv_params.groups;

  bool mod16_channels =
      C_per_group % 16 == 0 && (O_per_group <= 16 || O_per_group % 16 == 0);

  // Check if we can do implicit gemm but the channels are not divisible by 16
  // so we can pad and slice.
  //
  // We check it first because it doesn't need contiguous inputs and it needs
  // different output allocation.
  if (is_idil_one && !mod16_channels && conv_params.groups == 1) {
    return pad_and_slice_conv_3D_gpu(s, d, in_pre, wt_pre, out, conv_params);
  }

  // Allocate the output and ensure contiguous inputs
  out.set_data(allocator::malloc(out.nbytes()));
  auto in = ensure_row_contiguous(in_pre, d, s);
  auto wt = ensure_row_contiguous(wt_pre, d, s);

  // Perform the implicit gemm
  if (is_idil_one && mod16_channels) {
    return implicit_gemm_conv_3D_gpu(s, d, in, wt, out, conv_params);
  }

  // Explicit gemms where we unfold and do a matmul
  // (separate one for groups > 1)
  if (conv_params.groups > 1) {
    return explicit_gemm_conv_group_ND_gpu(s, d, in, wt, out, conv_params);
  }
  return explicit_gemm_conv_ND_gpu(s, d, in, wt, out, conv_params);
}

void winograd_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<2>& conv_params,
    std::vector<array>& copies_w) {
  Shape padded_shape = {
      conv_params.N,
      conv_params.iS[0] + 2 * conv_params.pad[0],
      conv_params.iS[1] + 2 * conv_params.pad[1],
      conv_params.C};

  padded_shape[1] = 6 * ((padded_shape[1] - 2 + 5) / 6) + 2;
  padded_shape[2] = 6 * ((padded_shape[2] - 2 + 5) / 6) + 2;

  array in_padded(std::move(padded_shape), in.dtype(), nullptr, {});

  // Fill with zeros
  array zero_arr = array(0, in.dtype());
  fill_gpu(zero_arr, in_padded, s);
  copies_w.push_back(zero_arr);

  // Pick input slice from padded
  size_t data_offset = conv_params.pad[0] * in_padded.strides()[1] +
      conv_params.pad[1] * in_padded.strides()[2];
  array in_padded_slice(in.shape(), in_padded.dtype(), nullptr, {});
  in_padded_slice.copy_shared_buffer(
      in_padded,
      in_padded.strides(),
      in_padded.flags(),
      in_padded_slice.size(),
      data_offset);

  // Copy input values into the slice
  copy_gpu_inplace(in, in_padded_slice, CopyType::GeneralGeneral, s);

  copies_w.push_back(in_padded_slice);
  copies_w.push_back(in_padded);

  MLXConvParams<2> conv_params_updated{
      /* const int  N = */ static_cast<int>(in_padded.shape(0)),
      /* const int  C = */ static_cast<int>(in_padded.shape(3)),
      /* const int  O = */ static_cast<int>(wt.shape(0)),
      /* const int iS[NDIM] = */
      {static_cast<int>(in_padded.shape(1)),
       static_cast<int>(in_padded.shape(2))},
      /* const int wS[NDIM] = */
      {static_cast<int>(wt.shape(1)), static_cast<int>(wt.shape(2))},
      /* const int oS[NDIM] = */
      {static_cast<int>(out.shape(1)), static_cast<int>(out.shape(2))},
      /* const int str[NDIM] = */ {1, 1},
      /* const int pad[NDIM] = */ {0, 0},
      /* const int kdil[NDIM] = */ {1, 1},
      /* const int idil[NDIM] = */ {1, 1},
      /* const size_t in_strides[NDIM + 2] = */
      {in_padded.strides()[0],
       in_padded.strides()[1],
       in_padded.strides()[2],
       in_padded.strides()[3]},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt.strides()[0], wt.strides()[1], wt.strides()[2], wt.strides()[3]},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides()[0], out.strides()[1], out.strides()[2], out.strides()[3]},
      /* const int groups = */ 1,
      /* const bool flip = */ false,
  };

  int O_c = conv_params.O;
  int C_c = conv_params.C;

  int N_tiles_n = conv_params.N;
  int N_tiles_h = (conv_params.oS[0] + 5) / 6;
  int N_tiles_w = (conv_params.oS[1] + 5) / 6;
  int N_tiles = N_tiles_n * N_tiles_h * N_tiles_w;

  // Do filter transform
  Shape filt_wg_shape = {8 * 8, conv_params.C, conv_params.O};
  array filt_wg(std::move(filt_wg_shape), wt.dtype(), nullptr, {});
  filt_wg.set_data(allocator::malloc(filt_wg.nbytes()));
  copies_w.push_back(filt_wg);
  {
    int bc = 32;
    int bo = 4;
    std::string kname;
    kname.reserve(32);
    concatenate(
        kname,
        "winograd_conv_2d_weight_transform_",
        type_to_name(out),
        "_bc",
        bc);
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname);
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
  inp_wg.set_data(allocator::malloc(inp_wg.nbytes()));
  copies_w.push_back(inp_wg);
  {
    int bc = 32;
    int wm = 2;
    int wn = 2;
    std::string kname;
    kname.reserve(32);
    concatenate(
        kname,
        "winograd_conv_2d_input_transform_",
        type_to_name(out),
        "_bc",
        bc);
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(in_padded, 0);
    compute_encoder.set_output_array(inp_wg, 1);

    compute_encoder.set_bytes(conv_params_updated, 2);

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(N_tiles_w, N_tiles_h, N_tiles_n);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Do batched gemm
  Shape out_wg_shape = {8 * 8, N_tiles, conv_params.O};
  array out_wg(std::move(out_wg_shape), in.dtype(), nullptr, {});
  out_wg.set_data(allocator::malloc(out_wg.nbytes()));
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
    std::string kname;
    kname.reserve(32);
    concatenate(
        kname,
        "winograd_conv_2d_output_transform_",
        type_to_name(out),
        "_bo",
        bc);
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(out_wg, 0);
    compute_encoder.set_output_array(out, 1);

    compute_encoder.set_bytes(conv_params_updated, 2);

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(N_tiles_w, N_tiles_h, N_tiles_n);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void depthwise_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<2>& conv_params) {
  std::string base_name;
  base_name.reserve(32);
  concatenate(base_name, "depthwise_conv_2d_", type_to_name(out));

  const int N = conv_params.N;
  const int ker_h = conv_params.wS[0];
  const int ker_w = conv_params.wS[1];
  const int str_h = conv_params.str[0];
  const int str_w = conv_params.str[1];
  const int tc = 8;
  const int tw = 8;
  const int th = 4;
  const bool do_flip = conv_params.flip;

  metal::MTLFCList func_consts = {
      {&ker_h, MTL::DataType::DataTypeInt, 00},
      {&ker_w, MTL::DataType::DataTypeInt, 01},
      {&str_h, MTL::DataType::DataTypeInt, 10},
      {&str_w, MTL::DataType::DataTypeInt, 11},
      {&th, MTL::DataType::DataTypeInt, 100},
      {&tw, MTL::DataType::DataTypeInt, 101},
      {&do_flip, MTL::DataType::DataTypeBool, 200},
  };

  // clang-format off
  std::string hash_name;
  hash_name.reserve(64);
  concatenate(
      hash_name,
      base_name,
  "_ker_h_", ker_h,
  "_ker_w_", ker_w,
  "_str_h_", str_h,
  "_str_w_", str_w,
  "_tgp_h_", th,
  "_tgp_w_", tw,
  "_do_flip_", do_flip ? 't' : 'n'); // clang-format on

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(base_name, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(wt, 1);
  compute_encoder.set_output_array(out, 2);

  compute_encoder.set_bytes(conv_params, 3);

  MTL::Size group_dims = MTL::Size(tc, tw, th);
  MTL::Size grid_dims = MTL::Size(
      conv_params.C / tc, conv_params.oS[1] / tw, (conv_params.oS[0] / th) * N);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void dispatch_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<2>& conv_params,
    std::vector<array>& copies) {
  bool is_stride_one = conv_params.str[0] == 1 && conv_params.str[1] == 1;
  bool is_kdil_one = conv_params.kdil[0] == 1 && conv_params.kdil[1] == 1;
  bool is_idil_one = conv_params.idil[0] == 1 && conv_params.idil[1] == 1;

  if (is_idil_one && conv_params.groups > 1) {
    const int C_per_group = conv_params.C / conv_params.groups;
    const int O_per_group = conv_params.O / conv_params.groups;

    if (C_per_group == 1 && O_per_group == 1 && is_kdil_one &&
        conv_params.wS[0] <= 7 && conv_params.wS[1] <= 7 &&
        conv_params.str[0] <= 2 && conv_params.str[1] <= 2 &&
        conv_params.oS[0] % 8 == 0 && conv_params.oS[1] % 8 == 0 &&
        conv_params.wt_strides[1] == conv_params.wS[1] &&
        conv_params.C % 16 == 0 && conv_params.C == conv_params.O) {
      return depthwise_conv_2D_gpu(s, d, in, wt, out, conv_params);
    }

    if ((C_per_group <= 4 || C_per_group % 16 == 0) &&
        (O_per_group <= 16 || O_per_group % 16 == 0)) {
      return implicit_gemm_conv_2D_gpu(s, d, in, wt, out, conv_params);
    } else {
      return explicit_gemm_conv_group_ND_gpu(s, d, in, wt, out, conv_params);
    }
  }

  // Direct to winograd conv
  bool inp_large =
      (conv_params.N * conv_params.iS[0] * conv_params.iS[1]) >= 4096;
  bool channels_large = (conv_params.C + conv_params.O) >= 256;
  bool out_large =
      (conv_params.N * conv_params.oS[0] * conv_params.oS[1]) >= 256;
  if (!conv_params.flip && is_stride_one && is_kdil_one && is_idil_one &&
      conv_params.wS[0] == 3 && conv_params.wS[1] == 3 &&
      conv_params.C % 32 == 0 && conv_params.O % 32 == 0 && inp_large &&
      channels_large) {
    return winograd_conv_2D_gpu(s, d, in, wt, out, conv_params, copies);
  }

  // Direct to implicit gemm conv
  if (is_idil_one && (conv_params.C <= 4 || conv_params.C % 16 == 0) &&
      (conv_params.O <= 16 || conv_params.O % 16 == 0)) {
    return implicit_gemm_conv_2D_gpu(s, d, in, wt, out, conv_params);
  }

  else if ((conv_params.C % 16 == 0 && conv_params.O % 16 == 0) || out_large) {
    return implicit_gemm_conv_2D_general_gpu(s, d, in, wt, out, conv_params);
  }

  // Direct to explicit gemm conv
  else {
    return explicit_gemm_conv_ND_gpu(s, d, in, wt, out, conv_params);
  }
}

void depthwise_conv_1D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array& out) {
  bool large = in.size() > INT32_MAX || in.data_size() > INT32_MAX;
  std::string base_name;
  base_name.reserve(32);
  concatenate(
      base_name,
      "depthwise_conv_1d_",
      large ? "_large" : "",
      type_to_name(out));

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(base_name);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto B = in.shape(0);
  auto Tout = out.shape(1);
  auto D = in.shape(2);
  auto K = wt.shape(1);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(wt, 1);
  compute_encoder.set_output_array(out, 2);
  if (large) {
    int64_t strides[3] = {in.strides(0), in.strides(1), in.strides(2)};
    compute_encoder.set_bytes(strides, 3, 3);

  } else {
    int strides[3] = {
        static_cast<int>(in.strides(0)),
        static_cast<int>(in.strides(1)),
        static_cast<int>(in.strides(2))};
    compute_encoder.set_bytes(strides, 3, 3);
  }

  compute_encoder.set_bytes(K, 4);
  auto group_dims = get_block_dims(D, Tout, B);
  MTL::Size grid_dims = MTL::Size(D, Tout, B);

  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void conv_1D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    int groups,
    bool flip,
    std::vector<array>& copies) {
  // Allocate space and ensure weights are contiguous
  out.set_data(allocator::malloc(out.nbytes()));
  auto in = ensure_row_contiguous(in_pre, d, s);
  auto wt = ensure_row_contiguous(wt_pre, d, s);

  bool is_idil_one = in_dilation[0] == 1;
  int C = in.shape(2);
  int O = wt.shape(0);
  // Fast path for fully separable 1D convolution
  if (is_idil_one && (groups == C) && groups == O && wt_strides[0] == 1 &&
      wt_dilation[0] == 1 && padding[0] == 0 && !flip) {
    depthwise_conv_1D_gpu(s, d, in, wt, out);
    return;
  }

  const int C_per_group = C / groups;
  const int O_per_group = O / groups;

  // Direct to implicit gemm conv
  if (is_idil_one && (C_per_group <= 4 || C_per_group % 16 == 0) &&
      (O_per_group <= 16 || O_per_group % 16 == 0)) {
    MLXConvParams<2> conv_params{
        /* const int  N = */ static_cast<int>(in.shape(0)),
        /* const int  C = */ C,
        /* const int  O = */ O,
        /* const int iS[NDIM] = */ {static_cast<int>(in.shape(1)), 1},
        /* const int wS[NDIM] = */ {static_cast<int>(wt.shape(1)), 1},
        /* const int oS[NDIM] = */ {static_cast<int>(out.shape(1)), 1},
        /* const int str[NDIM] = */ {wt_strides[0], 1},
        /* const int pad[NDIM] = */ {padding[0], 0},
        /* const int kdil[NDIM] = */ {wt_dilation[0], 1},
        /* const int idil[NDIM] = */ {in_dilation[0], 1},
        /* const size_t in_strides[NDIM + 2] = */
        {in.strides()[0], in.strides()[1], 0, in.strides()[2]},
        /* const size_t wt_strides[NDIM + 2] = */
        {wt.strides()[0], wt.strides()[1], 0, wt.strides()[2]},
        /* const size_t out_strides[NDIM + 2] = */
        {out.strides()[0], out.strides()[1], 0, out.strides()[2]},
        /* const int groups = */ groups,
        /* const bool flip = */ flip};

    dispatch_conv_2D_gpu(s, d, in, wt, out, conv_params, copies);
    return;
  }

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

void conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    const int groups,
    bool flip,
    std::vector<array>& copies) {
  // Allocate space and ensure weights are contiguous
  out.set_data(allocator::malloc(out.nbytes()));
  auto in = ensure_row_contiguous(in_pre, d, s);
  auto wt = ensure_row_contiguous(wt_pre, d, s);

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
  dispatch_conv_2D_gpu(s, d, in, wt, out, conv_params, copies);
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
    int groups,
    bool flip,
    std::vector<array>& copies) {
  // We will use the contiguous strides for the conv params because that is
  // what the rest of the code expects.
  constexpr int NDIM = 3;
  int64_t in_arr_strides[NDIM + 2];
  int64_t wt_arr_strides[NDIM + 2];
  in_arr_strides[NDIM + 1] = wt_arr_strides[NDIM + 1] = 1;
  for (int i = NDIM; i >= 0; i--) {
    in_arr_strides[i] = in_arr_strides[i + 1] * in.shape(i + 1);
    wt_arr_strides[i] = wt_arr_strides[i + 1] * wt.shape(i + 1);
  }

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
      {in_arr_strides[0],
       in_arr_strides[1],
       in_arr_strides[2],
       in_arr_strides[3],
       in_arr_strides[4]},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt_arr_strides[0],
       wt_arr_strides[1],
       wt_arr_strides[2],
       wt_arr_strides[3],
       wt_arr_strides[4]},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides(0),
       out.strides(1),
       out.strides(2),
       out.strides(3),
       out.strides(4)},
      /* const int groups = */ groups,
      /* const bool flip = */ flip,
  };
  return dispatch_conv_3D_gpu(s, d, in, wt, out, conv_params, copies);
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Intermediates that are put here will be added to the command encoder as
  // temporaries.
  std::vector<array> copies;

  // Some shortcuts for brevity
  const array& in = inputs[0];
  const array& wt = inputs[1];

  // 3D conv
  if (out.ndim() == 5) {
    conv_3D_gpu(
        s,
        d,
        in,
        wt,
        out,
        padding_lo_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        groups_,
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
        padding_lo_,
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
        padding_lo_,
        kernel_strides_,
        kernel_dilation_,
        input_dilation_,
        groups_,
        flip_,
        copies);
  }
  // Throw error
  else {
    throw std::invalid_argument(
        "[Convolution::eval_gpu] Only supports 1D, 2D or 3D convolutions.");
  }

  // Record copies
  if (!copies.empty()) {
    d.add_temporaries(std::move(copies), s.index);
  }
}

} // namespace mlx::core
