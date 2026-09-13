// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <cassert>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

// Above this the single pass kernel would need more than one read per thread,
// so switch to the looped variant.
constexpr int CROSS_ENTROPY_LOOPED_LIMIT = 4096;
constexpr int SIMD_SIZE = 32;
constexpr int N_READS = 4;

size_t ceil_simd_multiple(size_t n) {
  return SIMD_SIZE * ((n + SIMD_SIZE - 1) / SIMD_SIZE);
}

} // namespace

bool CrossEntropy::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void CrossEntropy::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2); // logits and targets
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);
  auto& out = outputs[0];

  auto ensure_row_contiguous = [&s, &compute_encoder](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    }
    array x_copy = contiguous_copy_gpu(x, s);
    compute_encoder.add_temporary(x_copy);
    return x_copy;
  };

  auto in = ensure_row_contiguous(inputs[0]); // [n_rows, V]
  auto target = ensure_row_contiguous(inputs[1]); // [n_rows]
  out.set_data(allocator::malloc(out.nbytes())); // [n_rows] in fp32

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  auto get_kernel = [&d, &in](bool looped) {
    std::string kernel_name = looped ? "looped_" : "block_";
    kernel_name += "cross_entropy_";
    kernel_name += type_to_name(in);
    return get_cross_entropy_kernel(d, kernel_name, in);
  };

  bool looped = axis_size > CROSS_ENTROPY_LOOPED_LIMIT;
  auto kernel = get_kernel(looped);
  size_t threadgroup_size = 0;
  if (!looped) {
    threadgroup_size = ceil_simd_multiple((axis_size + N_READS - 1) / N_READS);
    // The single pass kernel needs one thread per N_READS elements. If the
    // pipeline cannot hold that many threads, fall back to the looped variant.
    if (threadgroup_size > kernel->maxTotalThreadsPerThreadgroup()) {
      looped = true;
      kernel = get_kernel(true);
    }
  }
  if (looped) {
    threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
  }

  MTL::Size grid_dims(n_rows * threadgroup_size, 1, 1);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(target, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(axis_size, 3);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 4); // logits, targets, loss, cotangent
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);
  auto& out = outputs[0];

  auto ensure_row_contiguous = [&s, &compute_encoder](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    }
    array x_copy = contiguous_copy_gpu(x, s);
    compute_encoder.add_temporary(x_copy);
    return x_copy;
  };

  // The gradient has the same shape and type as the logits, so write it into
  // the logits buffer whenever that buffer is ours to reuse. The kernel reads
  // the target score into registers and barriers before overwriting anything.
  auto set_output = [&s, &out](const array& x) {
    if (x.flags().row_contiguous) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(allocator::malloc(out.nbytes()));
      }
      return x;
    }
    array x_copy = contiguous_copy_gpu(x, s);
    out.copy_shared_buffer(x_copy);
    return x_copy;
  };

  auto in = set_output(inputs[0]); // [n_rows, V]
  auto target = ensure_row_contiguous(inputs[1]); // [n_rows]
  auto loss = ensure_row_contiguous(inputs[2]); // [n_rows] fp32
  auto cotan = ensure_row_contiguous(inputs[3]); // [n_rows] fp32

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  std::string kernel_name = "vjp_cross_entropy_";
  kernel_name += type_to_name(in);
  auto kernel = get_cross_entropy_kernel(d, kernel_name, in);

  // The kernel loops over the row, so any threadgroup size is correct. Use
  // just enough threads to cover the row without oversubscribing short rows.
  size_t threadgroup_size = std::min(
      static_cast<size_t>(kernel->maxTotalThreadsPerThreadgroup()),
      ceil_simd_multiple((axis_size + N_READS - 1) / N_READS));

  MTL::Size grid_dims(n_rows * threadgroup_size, 1, 1);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(target, 1);
  compute_encoder.set_input_array(loss, 2);
  compute_encoder.set_input_array(cotan, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(axis_size, 5);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
