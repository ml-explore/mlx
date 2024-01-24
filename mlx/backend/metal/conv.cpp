// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/conv_params.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

void explicit_gemm_conv_1D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<1>& conv_params) {
  // Pad input
  std::vector<int> padded_shape = {
      conv_params.N, conv_params.iS[0] + 2 * conv_params.pad[0], conv_params.C};
  array in_padded(padded_shape, in.dtype(), nullptr, {});

  // Fill with zeros
  copy_gpu(array(0, in.dtype()), in_padded, CopyType::Scalar, s);

  // Pick input slice from padded
  size_t data_offset = conv_params.pad[0] * in_padded.strides()[1];
  array in_padded_slice(in.shape(), in_padded.dtype(), nullptr, {});
  in_padded_slice.copy_shared_buffer(
      in_padded,
      in_padded.strides(),
      in_padded.flags(),
      in_padded_slice.size(),
      data_offset);

  // Copy input values into the slice
  copy_gpu_inplace(in, in_padded_slice, CopyType::GeneralGeneral, s);

  // Make strided view
  std::vector<int> strided_shape = {
      conv_params.N, conv_params.oS[0], conv_params.wS[0], conv_params.C};

  std::vector<size_t> strided_strides = {
      in_padded.strides()[0],
      in_padded.strides()[1] * conv_params.str[0],
      in_padded.strides()[1],
      in_padded.strides()[2]};
  auto flags = in_padded.flags();

  array in_strided_view(strided_shape, in_padded.dtype(), nullptr, {});
  in_strided_view.copy_shared_buffer(
      in_padded, strided_strides, flags, in_strided_view.size(), 0);

  // Materialize strided view
  std::vector<int> strided_reshape = {
      conv_params.N * conv_params.oS[0], conv_params.wS[0] * conv_params.C};
  array in_strided(strided_reshape, in_strided_view.dtype(), nullptr, {});
  copy_gpu(in_strided_view, in_strided, CopyType::General, s);

  // Perform gemm
  std::vector<array> copies = {in_padded, in_strided};
  return steel_matmul(
      s,
      d,
      /*a = */ in_strided,
      /*b = */ wt,
      /*c = */ out,
      /*M = */ strided_reshape[0],
      /*N = */ conv_params.O,
      /*K = */ strided_reshape[1],
      /*batch_size_out = */ 1,
      /*a_cols = */ strided_reshape[1],
      /*b_cols = */ strided_reshape[1],
      /*a_transposed = */ false,
      /*b_transposed = */ true,
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
    const std::vector<int>& wt_dilation) {
  // Make conv params
  MLXConvParams<1> conv_params{
      /* const int  N = */ in.shape(0),
      /* const int  C = */ in.shape(2),
      /* const int  O = */ wt.shape(0),
      /* const int iS[NDIM] = */ {in.shape(1)},
      /* const int wS[NDIM] = */ {wt.shape(1)},
      /* const int oS[NDIM] = */ {out.shape(1)},
      /* const int str[NDIM] = */ {wt_strides[0]},
      /* const int pad[NDIM] = */ {padding[0]},
      /* const int dil[NDIM] = */ {wt_dilation[0]},
      /* const size_t in_strides[NDIM + 2] = */
      {in.strides()[0], in.strides()[1], in.strides()[2]},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt.strides()[0], wt.strides()[1], wt.strides()[2]},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides()[0], out.strides()[1], out.strides()[2]},
  };

  // Direct to explicit gemm conv
  if (wt_dilation[0] == 1) {
    explicit_gemm_conv_1D_gpu(s, d, in, wt, out, conv_params);
  }

  // Direct to fallback conv
  else {
    throw std::invalid_argument("[conv_1D_gpu] Dilation needs to be 1.");
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
  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder->setComputePipelineState(kernel);

  size_t n_pixels = conv_params.oS[0] * conv_params.oS[1];

  size_t grid_dim_x = (n_pixels + (tm * bm) - 1) / (tm * bm);
  size_t grid_dim_y = (conv_params.O + (tn * bn) - 1) / (tn * bn);
  size_t grid_dim_z = conv_params.N;

  MTL::Size group_dims = MTL::Size(bm, bn, 1);
  MTL::Size grid_dims = MTL::Size(grid_dim_x, grid_dim_y, grid_dim_z);

  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, wt, 1);
  set_array_buffer(compute_encoder, out, 2);

  compute_encoder->setBytes(&conv_params, sizeof(MLXConvParams<2>), 3);
  compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
}

void implicit_gemm_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params) {
  int bm = 32, bn = 32, bk = 16;
  int wm = 2, wn = 2;

  std::ostringstream kname;
  kname << "implicit_gemm_conv_2d_" << type_to_name(out) << "_bm" << bm << "_bn"
        << bn << "_bk" << bk << "_wm" << wm << "_wn" << wn;

  // Encode and dispatch kernel
  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder->setComputePipelineState(kernel);

  int implicit_M = conv_params.N * conv_params.oS[0] * conv_params.oS[1];
  int implicit_N = conv_params.O;
  int implicit_K = conv_params.wS[0] * conv_params.wS[1] * conv_params.C;

  size_t grid_dim_x = (implicit_N + bn - 1) / bn;
  size_t grid_dim_y = (implicit_M + bm - 1) / bm;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(grid_dim_x, grid_dim_y, 1);

  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, wt, 1);
  set_array_buffer(compute_encoder, out, 2);

  compute_encoder->setBytes(&conv_params, sizeof(MLXConvParams<2>), 3);
  compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
}

void explicit_gemm_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params) {
  // Pad input
  std::vector<int> padded_shape = {
      conv_params.N,
      conv_params.iS[0] + 2 * conv_params.pad[0],
      conv_params.iS[1] + 2 * conv_params.pad[1],
      conv_params.C};
  array in_padded(padded_shape, in.dtype(), nullptr, {});

  // Fill with zeros
  copy_gpu(array(0, in.dtype()), in_padded, CopyType::Scalar, s);

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

  // Make strided view
  std::vector<int> strided_shape = {
      conv_params.N,
      conv_params.oS[0],
      conv_params.oS[1],
      conv_params.wS[0],
      conv_params.wS[1],
      conv_params.C};

  std::vector<size_t> strided_strides = {
      in_padded.strides()[0],
      in_padded.strides()[1] * conv_params.str[0],
      in_padded.strides()[2] * conv_params.str[1],
      in_padded.strides()[1],
      in_padded.strides()[2],
      in_padded.strides()[3]};
  auto flags = in_padded.flags();

  array in_strided_view(strided_shape, in_padded.dtype(), nullptr, {});
  in_strided_view.copy_shared_buffer(
      in_padded, strided_strides, flags, in_strided_view.size(), 0);

  // Materialize strided view
  std::vector<int> strided_reshape = {
      conv_params.N * conv_params.oS[0] * conv_params.oS[1],
      conv_params.wS[0] * conv_params.wS[1] * conv_params.C};
  array in_strided(strided_reshape, in_strided_view.dtype(), nullptr, {});
  copy_gpu(in_strided_view, in_strided, CopyType::General, s);

  // Perform gemm
  std::vector<array> copies = {in_padded, in_strided};
  return steel_matmul(
      s,
      d,
      /*a = */ in_strided,
      /*b = */ wt,
      /*c = */ out,
      /*M = */ strided_reshape[0],
      /*N = */ conv_params.O,
      /*K = */ strided_reshape[1],
      /*batch_size_out = */ 1,
      /*a_cols = */ strided_reshape[1],
      /*b_cols = */ strided_reshape[1],
      /*a_transposed = */ false,
      /*b_transposed = */ true,
      /*copies = */ copies);
}

void winograd_conv_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params,
    std::vector<array>& copies_w) {
  std::vector<int> padded_shape = {
      conv_params.N,
      conv_params.iS[0] + 2 * conv_params.pad[0],
      conv_params.iS[1] + 2 * conv_params.pad[1],
      conv_params.C};

  padded_shape[1] = 6 * ((padded_shape[1] - 2 + 5) / 6) + 2;
  padded_shape[2] = 6 * ((padded_shape[2] - 2 + 5) / 6) + 2;

  array in_padded(padded_shape, in.dtype(), nullptr, {});

  // Fill with zeros
  array zero_arr = array(0, in.dtype());
  copy_gpu(zero_arr, in_padded, CopyType::Scalar, s);

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
  copies_w.push_back(zero_arr);

  MLXConvParams<2> conv_params_updated{
      /* const int  N = */ in_padded.shape(0),
      /* const int  C = */ in_padded.shape(3),
      /* const int  O = */ wt.shape(0),
      /* const int iS[NDIM] = */ {in_padded.shape(1), in_padded.shape(2)},
      /* const int wS[NDIM] = */ {wt.shape(1), wt.shape(2)},
      /* const int oS[NDIM] = */ {out.shape(1), out.shape(2)},
      /* const int str[NDIM] = */ {1, 1},
      /* const int pad[NDIM] = */ {0, 0},
      /* const int dil[NDIM] = */ {1, 1},
      /* const size_t in_strides[NDIM + 2] = */
      {in_padded.strides()[0],
       in_padded.strides()[1],
       in_padded.strides()[2],
       in_padded.strides()[3]},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt.strides()[0], wt.strides()[1], wt.strides()[2], wt.strides()[3]},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides()[0], out.strides()[1], out.strides()[2], out.strides()[3]},
  };

  int O_c = conv_params.O;
  int C_c = conv_params.C;

  int N_tiles_n = conv_params.N;
  int N_tiles_h = (conv_params.oS[0] + 5) / 6;
  int N_tiles_w = (conv_params.oS[1] + 5) / 6;
  int N_tiles = N_tiles_n * N_tiles_h * N_tiles_w;

  // Do filter transform
  std::vector<int> filt_wg_shape = {8 * 8, conv_params.C, conv_params.O};
  array filt_wg(filt_wg_shape, wt.dtype(), nullptr, {});
  filt_wg.set_data(allocator::malloc_or_wait(filt_wg.nbytes()));
  copies_w.push_back(filt_wg);
  {
    int bc = 32;
    int bo = 4;
    std::ostringstream kname;
    kname << "winograd_conv_2d_weight_transform_" << type_to_name(out) << "_bc"
          << bc;
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    set_array_buffer(compute_encoder, wt, 0);
    set_array_buffer(compute_encoder, filt_wg, 1);

    compute_encoder->setBytes(&C_c, sizeof(int), 2);
    compute_encoder->setBytes(&O_c, sizeof(int), 3);

    MTL::Size group_dims = MTL::Size(32, bo, 1);
    MTL::Size grid_dims = MTL::Size(O_c / bo, 1, 1);

    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  }

  // Do input transform
  std::vector<int> inp_wg_shape = {8 * 8, N_tiles, conv_params.C};
  array inp_wg(inp_wg_shape, in.dtype(), nullptr, {});
  inp_wg.set_data(allocator::malloc_or_wait(inp_wg.nbytes()));
  copies_w.push_back(inp_wg);
  {
    int bc = 32;
    int wm = 2;
    int wn = 2;
    std::ostringstream kname;
    kname << "winograd_conv_2d_input_transform_" << type_to_name(out) << "_bc"
          << bc;
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    set_array_buffer(compute_encoder, in_padded, 0);
    set_array_buffer(compute_encoder, inp_wg, 1);

    compute_encoder->setBytes(
        &conv_params_updated, sizeof(MLXConvParams<2>), 2);

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(N_tiles_w, N_tiles_h, N_tiles_n);

    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  }

  // Do batched gemm
  std::vector<int> out_wg_shape = {8 * 8, N_tiles, conv_params.O};
  array out_wg(out_wg_shape, in.dtype(), nullptr, {});
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
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    set_array_buffer(compute_encoder, out_wg, 0);
    set_array_buffer(compute_encoder, out, 1);

    compute_encoder->setBytes(
        &conv_params_updated, sizeof(MLXConvParams<2>), 2);

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(N_tiles_w, N_tiles_h, N_tiles_n);

    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
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
    std::vector<array>& copies) {
  // Make conv params
  MLXConvParams<2> conv_params{
      /* const int  N = */ in.shape(0),
      /* const int  C = */ in.shape(3),
      /* const int  O = */ wt.shape(0),
      /* const int iS[NDIM] = */ {in.shape(1), in.shape(2)},
      /* const int wS[NDIM] = */ {wt.shape(1), wt.shape(2)},
      /* const int oS[NDIM] = */ {out.shape(1), out.shape(2)},
      /* const int str[NDIM] = */ {wt_strides[0], wt_strides[1]},
      /* const int pad[NDIM] = */ {padding[0], padding[1]},
      /* const int dil[NDIM] = */ {wt_dilation[0], wt_dilation[1]},
      /* const size_t in_strides[NDIM + 2] = */
      {in.strides()[0], in.strides()[1], in.strides()[2], in.strides()[3]},
      /* const size_t wt_strides[NDIM + 2] = */
      {wt.strides()[0], wt.strides()[1], wt.strides()[2], wt.strides()[3]},
      /* const size_t out_strides[NDIM + 2] = */
      {out.strides()[0], out.strides()[1], out.strides()[2], out.strides()[3]},
  };

  // Direct to winograd conv
  if (conv_params.C % 32 == 0 && conv_params.O % 32 == 0 &&
      conv_params.C >= 64 && conv_params.O >= 64 && conv_params.wS[0] == 3 &&
      conv_params.wS[1] == 3 && conv_params.str[0] == 1 &&
      conv_params.str[1] == 1 && conv_params.dil[0] == 1 &&
      conv_params.dil[1] == 1) {
    winograd_conv_2D_gpu(s, d, in, wt, out, conv_params, copies);
  }

  // Direct to implicit gemm conv
  else if (conv_params.C % 32 == 0 && conv_params.O % 32 == 0) {
    implicit_gemm_conv_2D_gpu(s, d, in, wt, out, conv_params);
  }

  // Direct to explicit gemm conv
  else if (wt_dilation[0] == 1 && wt_dilation[1] == 1) {
    explicit_gemm_conv_2D_gpu(s, d, in, wt, out, conv_params);
  }

  // Direct to fallback conv
  else {
    slow_conv_2D_gpu(s, d, in, wt, out, conv_params);
  }
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

  // 2D conv
  if (out.ndim() == 4) {
    conv_2D_gpu(
        s, d, in, wt, out, padding_, kernel_strides_, kernel_dilation_, copies);
  }
  // 1D conv
  else if (out.ndim() == 3) {
    conv_1D_gpu(s, d, in, wt, out, padding_, kernel_strides_, kernel_dilation_);
  }
  // Throw error
  else {
    throw std::invalid_argument(
        "[Convolution::eval_gpu] Only supports 1D or 2D convolutions.");
  }

  // Clear copies
  if (copies.size() > 0) {
    auto command_buffer = d.get_command_buffer(s.index);
    command_buffer->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  }
}

} // namespace mlx::core
