// Copyright © 2025 Apple Inc.

#include <cmath>
#include <limits>

#include "mlx/types/half_types.h"

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
      if constexpr (is_floating_point_v<T>) {
        if (std::isnan(static_cast<float>(*output)) ||
            std::isnan(static_cast<float>(*input))) {
          *output = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        } else {
          *output = std::max(*output, *input);
        }
      } else {
        *output = std::max(*output, *input);
      }
      input++;
      output++;
    }
  }
};

template <typename T>
struct MinOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      if constexpr (is_floating_point_v<T>) {
        if (std::isnan(static_cast<float>(*output)) ||
            std::isnan(static_cast<float>(*input))) {
          *output = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        } else {
          *output = std::min(*output, *input);
        }
      } else {
        *output = std::min(*output, *input);
      }
      input++;
      output++;
    }
  }
};

} // namespace mlx::core::distributed::detail
