// Copyright © 2025 Apple Inc.

#include "mlx/distributed/nan_ops.h"

namespace mlx::core::distributed::detail {

template <typename T>
struct SumOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output += *input;
      input++;
      output++;
    }
  }
};

template <typename T>
struct MaxOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output = nan_aware_max(*output, *input);
      input++;
      output++;
    }
  }
};

template <typename T>
struct MinOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output = nan_aware_min(*output, *input);
      input++;
      output++;
    }
  }
};

} // namespace mlx::core::distributed::detail
