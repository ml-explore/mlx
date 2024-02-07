// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <iostream>
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

void slow_conv_transpose_2D_gpu(
    const Stream& s,
    metal::Device& d,
    const array& in,
    const array& wt,
    array out,
    const MLXConvParams<2>& conv_params) {
  int bm = 16, bn = 8;
  int tm = 4, tn = 4;

  std::ostringstream kname;
  kname << "naive_conv_transpose_2d_" << type_to_name(out) << "_bm" << bm
        << "_bn" << bn << "_tm" << tm << "_tn" << tn;

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

void conv_transpose_2D_gpu(
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

  slow_conv_transpose_2D_gpu(s, d, in, wt, out, conv_params);
}

} // namespace

void TransposeConvolution::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
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
    conv_transpose_2D_gpu(
        s, d, in, wt, out, padding_, kernel_strides_, kernel_dilation_, copies);
  }
  // 1D conv : TODO
  // else if (out.ndim() == 3) {
  //   conv_1D_gpu(s, d, in, wt, out, padding_, kernel_strides_,
  //   kernel_dilation_);
  // }
  // Throw error
  else {
    throw std::invalid_argument(
        "[TransposeConvolution::eval_gpu] Only supports 2D transpose convolutions.");
  }

  // Clear copies
  if (copies.size() > 0) {
    auto command_buffer = d.get_command_buffer(s.index);
    command_buffer->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  }
}

} // namespace mlx::core
