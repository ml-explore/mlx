// Copyright © 2023-2024 Apple Inc.

#include "mlx/array.h"
#include "mlx/primitives.h"

#define DEFAULT(primitive)                                                 \
  void primitive::eval_cpu(const std::vector<array>& inputs, array& out) { \
    primitive::eval(inputs, out);                                          \
  }

#define DEFAULT_MULTI(primitive)                                       \
  void primitive::eval_cpu(                                            \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    primitive::eval(inputs, outputs);                                  \
  }

namespace mlx::core {

DEFAULT(Reduce)
DEFAULT(Scan)

} // namespace mlx::core
