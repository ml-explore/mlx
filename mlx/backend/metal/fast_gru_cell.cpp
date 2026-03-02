// Copyright © 2024 Apple Inc.
// Fused GRU cell Metal backend. See Apple Metal docs:
// https://developer.apple.com/documentation/metal

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

void FastGruCell::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastGruCell::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& input_proj = inputs[0];
  const array& hidden_proj = inputs[1];
  const array& hidden_prev = inputs[2];
  const bool has_bhn = (inputs.size() == 4);
  array& out = outputs[0];

  if (input_proj.dtype() != float32 && input_proj.dtype() != bfloat16) {
    outputs = fallback_(inputs);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  std::vector<array> copies;
  auto copy_if_needed = [&copies, &s](const array& a) -> const array& {
    if (a.flags().row_contiguous)
      return a;
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };
  const array& in_proj = copy_if_needed(input_proj);
  const array& hid_proj = copy_if_needed(hidden_proj);
  const array& hid_prev = copy_if_needed(hidden_prev);
  const array* bhn_ptr = has_bhn ? &copy_if_needed(inputs[3]) : nullptr;

  size_t batch_size = in_proj.shape(0);
  size_t hidden_size = hid_prev.shape(1);
  uint32_t h_quads = (static_cast<uint32_t>(hidden_size) + 3) / 4;
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * h_quads;

  std::string kname;
  if (has_bhn) {
    kname = (in_proj.dtype() == bfloat16) ? "gru_cell_fused_bfloat16_bias"
                                          : "gru_cell_fused_float_bias";
  } else {
    kname = (in_proj.dtype() == bfloat16) ? "gru_cell_fused_bfloat16"
                                          : "gru_cell_fused_float";
  }

  auto& enc = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(in_proj, 0);
  enc.set_input_array(hid_proj, 1);
  enc.set_input_array(hid_prev, 2);
  if (has_bhn) {
    enc.set_input_array(*bhn_ptr, 3);
    enc.set_output_array(out, 4);
  } else {
    enc.set_output_array(out, 3);
  }
  uint32_t bs = static_cast<uint32_t>(batch_size);
  uint32_t hs = static_cast<uint32_t>(hidden_size);
  uint32_t bytes_base = has_bhn ? 5u : 4u;
  enc.set_bytes(bs, bytes_base);
  enc.set_bytes(hs, bytes_base + 1);

  constexpr uint32_t threads_per_group = 512;
  uint32_t num_groups =
      (total_threads + threads_per_group - 1) / threads_per_group;
  MTL::Size grid_dims(num_groups, 1, 1);
  MTL::Size group_dims(threads_per_group, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

bool FastGruCell::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

} // namespace mlx::core::fast
