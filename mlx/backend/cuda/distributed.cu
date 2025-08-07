// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/distributed/primitives.h"
#include "mlx/primitives.h"

#include <cassert>

namespace mlx::core {
namespace distributed {
void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& input = inputs[0];
  auto& output = outputs[0];

  auto& encoder = cu::get_command_encoder(stream());

  if (input.is_donatable()) {
    output.copy_shared_buffer(input);
  } else {
    output.set_data(allocator::malloc(output.nbytes()));
  }

  encoder.set_input_array(input);
  encoder.set_output_array(output);

  auto capture = encoder.capture_context();
  auto& s = stream();

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
} // namespace distributed
} // namespace mlx::core