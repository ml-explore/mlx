// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Just a fall-back to the original tape for now
  std::unordered_map<uintptr_t, array> trace_to_real;
  for (int i = 0; i < inputs.size(); ++i) {
    trace_to_real.insert({inputs_[i].id(), inputs[i]});
  }
  for (int i = 0; i < outputs.size(); ++i) {
    trace_to_real.insert({outputs_[i].id(), outputs[i]});
  }

  for (auto& a : tape_) {
    std::vector<array> p_inputs;
    for (auto& in : a.inputs()) {
      p_inputs.push_back(trace_to_real.at(in.id()));
    }
    // If a is an output get it from the map, otherwise create it
    // NB this is safe as long as no multi-output sub primitves are allowed
    // in Compiled
    std::vector<array> p_outputs;
    if (auto it = trace_to_real.find(a.id()); it != trace_to_real.end()) {
      p_outputs.push_back(it->second);
    } else {
      p_outputs.push_back(array(a.shape(), a.dtype(), a.primitive_ptr(), {}));
      trace_to_real.insert({a.id(), p_outputs[0]});
    }
    a.primitive().eval_gpu(p_inputs, p_outputs);
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);
  command_buffer->addCompletedHandler(
      [trace_to_real](MTL::CommandBuffer*) mutable {});
}

} // namespace mlx::core
