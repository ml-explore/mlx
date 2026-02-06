// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/conv/conv.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

// Forward declaration of gemm_conv functions
void gemm_conv(
    rocm::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    bool flip,
    Stream s);

void gemm_grouped_conv(
    rocm::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s);

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  array in = inputs[0];
  array wt = inputs[1];

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  // Ensure inputs are contiguous
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in);
  }
  if (!wt.flags().row_contiguous) {
    wt = contiguous_copy_gpu(wt, s);
    encoder.add_temporary(wt);
  }

  // Use GEMM-based convolution
  if (groups_ == 1) {
    gemm_conv(
        encoder,
        in,
        wt,
        out,
        kernel_strides_,
        padding_lo_,
        kernel_dilation_,
        input_dilation_,
        flip_,
        s);
  } else {
    gemm_grouped_conv(
        encoder,
        in,
        wt,
        out,
        kernel_strides_,
        padding_lo_,
        kernel_dilation_,
        input_dilation_,
        groups_,
        flip_,
        s);
  }
}

} // namespace mlx::core
