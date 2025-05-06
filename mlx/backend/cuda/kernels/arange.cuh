// Copyright © 2025 Apple Inc.

namespace mlx::core::cu {

template <typename T>
struct Arange {
  const T start;
  const T step;

  __device__ T operator()(uint32_t i) const {
    return start + i * step;
  }
};

} // namespace mlx::core::cu
