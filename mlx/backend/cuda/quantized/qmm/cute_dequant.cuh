// Copyright © 2026 Apple Inc.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

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
