// Copyright © 2025 Apple Inc.

namespace mlx::core {

namespace cu {

template <int bits, int wsize = 8>
inline constexpr __device__ short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr __device__ short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T>
__device__ __forceinline__ void abs_max_x2(T& out, const T& x1, const T& x2) {
  if constexpr (
      (std::is_same<T, __nv_bfloat162>::value) ||
      (std::is_same<T, __half2>::value)) {
    T a = x1;
    T b = x2;
    out = __hmax2(__habs2(a), __habs2(b));
  } else if constexpr (std::is_same<T, float2>::value) {
    float2 a = x1;
    float2 b = x2;
    out.x = fmaxf(fabsf(a.x), fabsf(b.x));
    out.y = fmaxf(fabsf(a.y), fabsf(b.y));
  }
}

} // namespace cu

template <typename F>
void dispatch_groups(int group_size, F&& f) {
  switch (group_size) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    case 128:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

template <typename F>
void dispatch_bits(int bits, F&& f) {
  switch (bits) {
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 3:
      f(std::integral_constant<int, 3>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
    case 5:
      f(std::integral_constant<int, 5>{});
      break;
    case 6:
      f(std::integral_constant<int, 6>{});
      break;
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
  }
}

} // namespace mlx::core
