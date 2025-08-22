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

  auto set_input_output =
      [s = stream()](const array& in, array& out) -> std::pair<array, array> {
    if (!in.flags().row_contiguous) {
      copy_gpu(in, out, CopyType::General, s);
      return {out, out};
    } else if (in.is_donatable()) {
      out.copy_shared_buffer(in);
      return {in, out};
    } else {
      return {in, out};
    }
  };

  auto [input, output] = set_input_output(inputs[0], outputs[0]);

  auto& encoder = cu::get_command_encoder(stream());
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
} // namespace mlx::core::distributed
