// Copyright © 2023-2024 Apple Inc.

#include <queue>

#include "mlx/primitives.h"

namespace mlx::core {

// Build the real tape
std::pair<std::queue<array>, std::vector<array>> trace_to_real(
    const std::vector<array>& trace_tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs) {
  std::unordered_map<uintptr_t, array> trace_to_real;
  for (int i = 0; i < inputs.size(); ++i) {
    trace_to_real.insert({trace_inputs[i].id(), inputs[i]});
  }
  std::queue<array> tape;
  for (auto& a : trace_tape) {
    // Find real inputs
    std::vector<array> real_inputs;
    for (auto& in : a.inputs()) {
      real_inputs.push_back(trace_to_real.at(in.id()));
    }
    tape.push(
        array(a.shape(), a.dtype(), a.primitive_ptr(), std::move(real_inputs)));
    trace_to_real.insert({a.id(), tape.back()});
  }

  std::vector<array> outputs;
  for (auto& o : trace_outputs) {
    outputs.push_back(trace_to_real.at(o.id()));
  }
  return {tape, outputs};
}

void Compiled::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Make the a real tape from the tracers
  auto [tape, real_outputs] = trace_to_real(tape_, inputs_, outputs_, inputs);

  // Run the tape
  while (!tape.empty()) {
    auto a = std::move(tape.front());
    tape.pop();
    auto outputs = a.outputs();
    a.primitive().eval_cpu(a.inputs(), outputs);
    a.detach();
  }

  // Copy results into outputs
  for (int o = 0; o < real_outputs.size(); ++o) {
    outputs[o].copy_shared_buffer(real_outputs[o]);
  }
}

} // namespace mlx::core
