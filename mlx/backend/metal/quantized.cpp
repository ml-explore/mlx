// Copyright © 2023 Apple Inc.

#include <cassert>
#include <iostream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];

  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };
  auto [x_transposed, x_cols, x] = check_transpose(x_pre);
  auto [w_transposed, w_cols, w] = check_transpose(w_pre);
  auto [scales_transposed, scales_cols, scales] = check_transpose(scales_pre);
  auto [biases_transposed, biases_cols, biases] = check_transpose(biases_pre);

  if (!w_transposed) {
    throw std::runtime_error("The quantized weight should be transposed.");
  }

  if (x_transposed || scales_transposed || biases_transposed) {
    throw std::runtime_error("x, scales and biases should be row contiguous.");
  }

  int D = x.shape(-1);
  int B = x.size() / D;

  // Route to the qmv kernel
  if (B == 1) {
    std::ostringstream kname;
    kname << "qmv_" << (w_transposed ? "n_" : "t_") << type_to_name(out)
          << "_gs_" << group_size_ << "_b_" << bits_;

    // Encode and dispatch kernel
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    int O = w.size() / w_cols;

    int bo = 32;
    int bd = 32;
    MTL::Size group_dims = MTL::Size(bd, bo, 1);
    MTL::Size grid_dims = MTL::Size(1, O / bo, B);

    set_array_buffer(compute_encoder, w, 0);
    set_array_buffer(compute_encoder, scales, 1);
    set_array_buffer(compute_encoder, biases, 2);
    set_array_buffer(compute_encoder, x, 3);
    set_array_buffer(compute_encoder, out, 4);
    compute_encoder->setBytes(&D, sizeof(int), 5);
    compute_encoder->setBytes(&O, sizeof(int), 6);

    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  }

  // Route to the qmm kernel
  else {
    std::ostringstream kname;
    kname << "qmm_" << (w_transposed ? "t_" : "n_") << type_to_name(out)
          << "_gs_" << group_size_ << "_b_" << bits_;

    // Encode and dispatch kernel
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    int O = w.size() / w_cols;

    int wn = 2;
    int wm = 2;
    int bm = 32;
    int bn = 32;
    int bk = 64;
    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(O / bn, (B + bm - 1) / bm, 1);

    set_array_buffer(compute_encoder, x, 0);
    set_array_buffer(compute_encoder, w, 1);
    set_array_buffer(compute_encoder, scales, 2);
    set_array_buffer(compute_encoder, biases, 3);
    set_array_buffer(compute_encoder, out, 4);
    compute_encoder->setBytes(&B, sizeof(int), 5);
    compute_encoder->setBytes(&O, sizeof(int), 6);
    compute_encoder->setBytes(&D, sizeof(int), 7);

    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  }

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
