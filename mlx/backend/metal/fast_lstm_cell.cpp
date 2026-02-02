// Copyright © 2024 Apple Inc.
// Fused LSTM cell Metal backend.

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

void FastLSTMCell::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastLSTMCell::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& input_proj = inputs[0];
  const array& hidden_proj = inputs[1];
  const array& cell_prev = inputs[2];
  const array& hidden_prev = inputs[3];
  array& out_cell = outputs[0];
  array& out_hidden = outputs[1];

  if (input_proj.dtype() != float32 && input_proj.dtype() != bfloat16) {
    outputs = fallback_(inputs);
    return;
  }

  // Allocate output buffers on GPU (required before set_output_array)
  out_cell.set_data(allocator::malloc(out_cell.nbytes()));
  out_hidden.set_data(allocator::malloc(out_hidden.nbytes()));

  std::vector<array> copies;
  auto copy_if_needed = [&copies, &s](const array& a) -> const array& {
    if (a.flags().row_contiguous)
      return a;
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };
  const array& in_proj = copy_if_needed(input_proj);
  const array& hid_proj = copy_if_needed(hidden_proj);
  const array& c_prev = copy_if_needed(cell_prev);
  const array& hid_prev = copy_if_needed(hidden_prev);

  size_t batch_size = in_proj.shape(0);
  size_t hidden_size = c_prev.shape(1);
  uint32_t h_quads = (static_cast<uint32_t>(hidden_size) + 3) / 4;
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * h_quads;

  std::string kname = (in_proj.dtype() == bfloat16) ? "lstm_cell_fused_bfloat16"
                                                    : "lstm_cell_fused_float";

  auto& enc = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(in_proj, 0);
  enc.set_input_array(hid_proj, 1);
  enc.set_input_array(c_prev, 2);
  enc.set_input_array(hid_prev, 3);
  enc.set_output_array(out_cell, 4);
  enc.set_output_array(out_hidden, 5);
  uint32_t bs = static_cast<uint32_t>(batch_size);
  uint32_t hs = static_cast<uint32_t>(hidden_size);
  enc.set_bytes(bs, 6);
  enc.set_bytes(hs, 7);

  constexpr uint32_t threads_per_group = 512;
  uint32_t num_groups =
      (total_threads + threads_per_group - 1) / threads_per_group;
  MTL::Size grid_dims(num_groups, 1, 1);
  MTL::Size group_dims(threads_per_group, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

bool FastLSTMCell::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

} // namespace mlx::core::fast
