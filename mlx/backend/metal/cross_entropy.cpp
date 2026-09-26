// Copyright © 2026 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

// Threadgroup shrinks with the axis so a small vocabulary needs one iteration.
inline std::pair<MTL::Size, MTL::Size> cross_entropy_dims(
    MTL::ComputePipelineState* kernel,
    int axis_size,
    int n_rows) {
  constexpr size_t simd_size = 32;
  size_t tg_max =
      (kernel->maxTotalThreadsPerThreadgroup() / simd_size) * simd_size;
  size_t needed =
      (axis_size + CROSS_ENTROPY_N_READS - 1) / CROSS_ENTROPY_N_READS;
  size_t threadgroup_size = std::clamp(
      ((needed + simd_size - 1) / simd_size) * simd_size, simd_size, tg_max);
  return {
      MTL::Size(n_rows * threadgroup_size, 1, 1),
      MTL::Size(threadgroup_size, 1, 1)};
}

array ensure_row_contiguous(
    const array& x,
    metal::CommandEncoder& encoder,
    const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  encoder.add_temporary(x_copy);
  return x_copy;
}

} // namespace

bool CrossEntropy::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void CrossEntropy::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& in_pre = inputs[0];
  auto& out = outputs[0];
  if (in_pre.size() == 0) {
    throw std::invalid_argument("[cross_entropy] Received empty array.");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto in = ensure_row_contiguous(in_pre, compute_encoder, s);
  auto targets = ensure_row_contiguous(inputs[1], compute_encoder, s);
  out.set_data(allocator::malloc(out.nbytes()));

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  std::string kernel_name = "cross_entropy_" + type_to_name(in);
  auto kernel = get_cross_entropy_kernel(d, kernel_name, in);
  auto [grid_dims, group_dims] = cross_entropy_dims(kernel, axis_size, n_rows);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(targets, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(axis_size, 3);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& in_pre = inputs[0];
  auto& out = outputs[0];
  if (in_pre.size() == 0) {
    throw std::invalid_argument("[cross_entropy] Received empty array.");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  bool donate_in = in_pre.is_donatable() || !in_pre.flags().row_contiguous;
  auto in = ensure_row_contiguous(in_pre, compute_encoder, s);
  auto targets = ensure_row_contiguous(inputs[1], compute_encoder, s);
  auto loss = ensure_row_contiguous(inputs[2], compute_encoder, s);
  auto cotan = ensure_row_contiguous(inputs[3], compute_encoder, s);
  if (donate_in) {
    out.copy_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  std::string kernel_name = "cross_entropy_vjp_" + type_to_name(in);
  auto kernel = get_cross_entropy_kernel(d, kernel_name, in);
  auto [grid_dims, group_dims] = cross_entropy_dims(kernel, axis_size, n_rows);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(targets, 1);
  compute_encoder.set_input_array(loss, 2);
  compute_encoder.set_input_array(cotan, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(axis_size, 5);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
