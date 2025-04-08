// Copyright © 2025 Apple Inc.

namespace mlx::core::mxcuda {

template <typename T>
struct Arange {
  const T start;
  const T step;

  __device__ T operator()(uint32_t i) const {
    return start + i * step;
  }
};

template <typename T>
inline Arange<T> make_arange(T start, T next) {
  return Arange<T>{start, static_cast<T>(next - start)};
}

} // namespace mlx::core::mxcuda
