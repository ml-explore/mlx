// Copyright © 2023 Apple Inc.

#include <cassert>

#include <simd/vector.h>
#include <vecLib/vDSP.h>

#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

void Convolution::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);

  // TODO: Add accelerate based optimizations for CPU conv
}

} // namespace mlx::core
