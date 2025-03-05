// Copyright © 2025 Apple Inc.
#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::cpu {

void eval(array& arr) {
  auto s = arr.primitive().stream();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_cpu(arr.inputs(), outputs);
  }
}

void finalize(
    Stream s,
    std::unordered_set<std::shared_ptr<array::Data>> retain_buffers) {
  auto& encoder = cpu::get_command_encoder(s);
  encoder.dispatch([s,
                    buffers = std::move(retain_buffers),
                    temps = std::move(encoder.temporaries())]() {});
}

} // namespace mlx::core::cpu
