// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/distributed/primitives.h"
#include "mlx/primitives.h"

#include <cassert>

namespace mlx::core::distributed {
void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  auto set_input_output = [&](const array& in,
                              array& out) -> std::pair<array, array> {
    if (!in.flags().row_contiguous) {
      copy_gpu(in, out, CopyType::General, s);
      return {out, out};
    } else if (in.is_donatable()) {
      out.copy_shared_buffer(in);
      return {in, out};
    } else {
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
      return {in, out};
    }
  };

  auto [input, output] = set_input_output(inputs[0], outputs[0]);

  encoder.set_input_array(input);
  encoder.set_output_array(output);

  auto capture = encoder.capture_context();

  switch (reduce_type_) {
    case Sum:
      distributed::detail::all_sum(group(), input, output, s);
      break;
    case Max:
      distributed::detail::all_max(group(), input, output, s);
      break;
    case Min:
      distributed::detail::all_min(group(), input, output, s);
      break;
    default:
      throw std::runtime_error(
          "Only all reduce sum, max, and min are supported.");
  }
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  auto ensure_contiguous = [&s, &encoder](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      encoder.add_temporary(x_copy);
      return x_copy;
    }
  };

  auto input = ensure_contiguous(inputs[0]);
  outputs[0].set_data(cu::malloc_async(outputs[0].nbytes(), encoder));

  encoder.set_input_array(input);
  encoder.set_output_array(outputs[0]);

  auto capture = encoder.capture_context();
  distributed::detail::all_gather(group(), input, outputs[0], s);
}

void ReduceScatter::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  auto ensure_contiguous = [&s, &encoder](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      encoder.add_temporary(x_copy);
      return x_copy;
    }
  };

  auto input = ensure_contiguous(inputs[0]);
  outputs[0].set_data(cu::malloc_async(outputs[0].nbytes(), encoder));

  encoder.set_input_array(input);
  encoder.set_output_array(outputs[0]);

  auto capture = encoder.capture_context();

  switch (reduce_type_) {
    case Sum:
      distributed::detail::sum_scatter(group(), input, outputs[0], s);
      break;
    default:
      throw std::runtime_error("Only sum scatter is supported. ");
  }
}
} // namespace mlx::core::distributed
