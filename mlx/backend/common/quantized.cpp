// Copyright © 2023 Apple Inc.

#include "mlx/primitives.h"

namespace mlx::core {

void QuantizedMatmul::eval(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // TODO: Implement the quant matmul
}

} // namespace mlx::core
