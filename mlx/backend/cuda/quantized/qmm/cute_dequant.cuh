// Copyright © 2026 Apple Inc.

#pragma once

#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cuda/std/array>

namespace cutlass {

using uint3b_t = integer_subbyte<3, false>;
using uint5b_t = integer_subbyte<5, false>;

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint3b_t, N, Round> {
  static_assert(N % 8 == 0);

  using result_type = Array<T, N>;
  using source_type = Array<uint3b_t, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      auto* s = s_base + i * 3;
      result[i * 8] = T(s[0] & 0x07);
      result[i * 8 + 1] = T((s[0] & 0x38) >> 3);
      result[i * 8 + 2] = T((s[0] & 0xc0) >> 6) + T((s[1] & 0x01) << 2);
      result[i * 8 + 3] = T((s[1] & 0x0e) >> 1);
      result[i * 8 + 4] = T((s[1] & 0x70) >> 4);
      result[i * 8 + 5] = T((s[1] & 0x80) >> 7) + T((s[2] & 0x03) << 1);
      result[i * 8 + 6] = T((s[2] & 0x1c) >> 2);
      result[i * 8 + 7] = T((s[2] & 0xe0) >> 5);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const {
    return convert(s);
  }
};

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint5b_t, N, Round> {
  static_assert(N % 8 == 0);

  using result_type = Array<T, N>;
  using source_type = Array<uint5b_t, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      auto* s = s_base + i * 5;
      result[i * 8] = T(s[0] & 0x1f);
      result[i * 8 + 1] = T((s[0] & 0xe0) >> 5) + T((s[1] & 0x03) << 3);
      result[i * 8 + 2] = T((s[1] & 0x7c) >> 2);
      result[i * 8 + 3] = T((s[1] & 0x80) >> 7) + T((s[2] & 0x0f) << 1);
      result[i * 8 + 4] = T((s[2] & 0xf0) >> 4) + T((s[3] & 0x01) << 4);
      result[i * 8 + 5] = T((s[3] & 0x3e) >> 1);
      result[i * 8 + 6] = T((s[3] & 0xc0) >> 6) + T((s[4] & 0x07) << 2);
      result[i * 8 + 7] = T((s[4] & 0xf8) >> 3);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const {
    return convert(s);
  }
};

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint6b_t, N, Round> {
  static_assert(N % 4 == 0);

  using result_type = Array<T, N>;
  using source_type = Array<uint6b_t, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      auto* s = s_base + i * 3;
      result[i * 4] = T(s[0] & 0x3f);
      result[i * 4 + 1] = T((s[0] >> 6) & 0x03) + T((s[1] & 0x0f) << 2);
      result[i * 4 + 2] = T((s[1] >> 4) & 0x0f) + T((s[2] & 0x03) << 4);
      result[i * 4 + 3] = T((s[2] >> 2) & 0x3f);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const {
    return convert(s);
  }
};

} // namespace cutlass

namespace cute {

// Required by tiled copy for 3/5/6-bit weights.
struct uint24_t {
  cuda::std::array<std::uint8_t, 3> bytes;
};
struct uint40_t {
  cuda::std::array<std::uint8_t, 5> bytes;
};
struct uint48_t {
  cuda::std::array<std::uint8_t, 6> bytes;
};

template <>
struct uint_bit<24> {
  using type = uint24_t;
};
template <>
struct uint_bit<40> {
  using type = uint40_t;
};
template <>
struct uint_bit<48> {
  using type = uint48_t;
};

} // namespace cute

namespace cutlass_gemm {

// Whether the quant type is affine quantization.
template <typename Quant>
constexpr bool quant_has_bias_v = !cutlass::has_negative_zero_v<Quant>;

// Dequantize CuTe tensors with out = w * s + z.
__device__ __forceinline__ void
cute_vectorized_dequant(auto w, auto s, auto z, auto out) {
  using namespace cute;
  using Element = typename decltype(out)::value_type;
  using Quant = typename decltype(w)::value_type;
  // Scale must be one element.
  CUTE_STATIC_ASSERT_V(cosize(s.layout()) == Int<1>{});
  CUTE_STATIC_ASSERT_V(cosize(z.layout()) == Int<1>{});
  // Quant must be contiguous.
  auto layout = coalesce(w.layout());
  CUTE_STATIC_ASSERT_V(stride(layout) == Int<1>{});
  // Use cutlass for conversions.
  constexpr int N = size(layout);
  auto& w_vec = *(reinterpret_cast<const cutlass::Array<Quant, N>*>(
      raw_pointer_cast(w.data())));
  Element scale{s[0]};
  cutlass::NumericArrayConverter<Element, Quant, N> converter;
  auto w_dq = converter(w_vec) * scale;
  if constexpr (quant_has_bias_v<Quant>) {
    Element zero_point{z[0]};
    w_dq = w_dq + zero_point;
  }
  copy(make_tensor(make_rmem_ptr<Element>(&w_dq), out.layout()), out);
}

} // namespace cutlass_gemm
