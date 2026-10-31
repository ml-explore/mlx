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
inline size_t cross_entropy_threadgroup_size(
    MTL::ComputePipelineState* kernel,
    int axis_size) {
  constexpr size_t simd_size = 32;
  size_t tg_max =
      (kernel->maxTotalThreadsPerThreadgroup() / simd_size) * simd_size;
  size_t needed =
      (axis_size + CROSS_ENTROPY_N_READS - 1) / CROSS_ENTROPY_N_READS;
  return std::clamp(
      ((needed + simd_size - 1) / simd_size) * simd_size, simd_size, tg_max);
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
  int64_t n_rows = in.data_size() / axis_size;

  std::string kernel_name = "cross_entropy_" + type_to_name(in);
  auto kernel = get_cross_entropy_kernel(d, kernel_name, in);
  size_t threadgroup_size = cross_entropy_threadgroup_size(kernel, axis_size);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(axis_size, 3);
  const int64_t row_bytes = int64_t(axis_size) * in.itemsize();
  dispatch_threadgroups_in_slices(
      compute_encoder, n_rows, threadgroup_size, [&](int64_t row) {
        compute_encoder.set_input_array(in, 0, row * row_bytes);
        compute_encoder.set_input_array(targets, 1, row * targets.itemsize());
        compute_encoder.set_output_array(out, 2, row * out.itemsize());
      });
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  array in = inputs[0];
  array& out = outputs[0];
  if (in.size() == 0) {
    throw std::invalid_argument("[cross_entropy] Received empty array.");
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  bool donate_in = in.is_donatable();
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_gpu(in, s);
    donate_in = true;
  }
  auto targets = ensure_row_contiguous(inputs[1], compute_encoder, s);
  auto loss = ensure_row_contiguous(inputs[2], compute_encoder, s);
  auto cotan = ensure_row_contiguous(inputs[3], compute_encoder, s);
  if (donate_in) {
    out.copy_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  int axis_size = in.shape().back();
  int64_t n_rows = in.data_size() / axis_size;

  std::string kernel_name = "cross_entropy_vjp_" + type_to_name(in);
  auto kernel = get_cross_entropy_kernel(d, kernel_name, in);
  size_t threadgroup_size = cross_entropy_threadgroup_size(kernel, axis_size);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(axis_size, 5);
  const int64_t row_bytes = int64_t(axis_size) * in.itemsize();
  dispatch_threadgroups_in_slices(
      compute_encoder, n_rows, threadgroup_size, [&](int64_t row) {
        compute_encoder.set_input_array(in, 0, row * row_bytes);
        compute_encoder.set_input_array(targets, 1, row * targets.itemsize());
        compute_encoder.set_input_array(loss, 2, row * loss.itemsize());
        compute_encoder.set_input_array(cotan, 3, row * cotan.itemsize());
        compute_encoder.set_output_array(out, 4, row * row_bytes);
      });
}

} // namespace mlx::core::fast
