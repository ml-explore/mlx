// Copyright © 2025 Apple Inc.

#pragma once

#include <cuComplex.h>
#include <thrust/iterator/transform_iterator.h>

namespace mlx::core::cu {

// An op that does static_cast, with custom conversions for some types.
template <typename SrcT, typename DstT, typename = void>
struct CastOp {
  static constexpr bool is_castable = cuda::std::is_convertible_v<SrcT, DstT>;

  __device__ DstT operator()(SrcT x) {
    return static_cast<DstT>(x);
  }
};

// Converting a complex number to real number discards the imaginary part.
template <typename DstT>
struct CastOp<
    cuComplex,
    DstT,
    cuda::std::enable_if_t<!cuda::std::is_same_v<cuComplex, DstT>>> {
  static constexpr bool is_castable = cuda::std::is_convertible_v<float, DstT>;

  __device__ DstT operator()(cuComplex x) {
    static_assert(!cuda::std::is_same_v<cuComplex, DstT>);
    return static_cast<DstT>(cuCrealf(x));
  }
};

// Allow converting a real number to complex number.
template <typename SrcT>
struct CastOp<
    SrcT,
    cuComplex,
    cuda::std::enable_if_t<!cuda::std::is_same_v<SrcT, cuComplex>>> {
  static constexpr bool is_castable = cuda::std::is_convertible_v<SrcT, float>;

  __device__ cuComplex operator()(SrcT x) {
    static_assert(!cuda::std::is_same_v<SrcT, cuComplex>);
    return cuComplex{static_cast<float>(x), 0};
  }
};

template <typename SrcT, typename DstT>
struct CastOp<
    SrcT,
    DstT,
    cuda::std::enable_if_t<cuda::std::is_same_v<SrcT, DstT>>> {
  static constexpr bool is_castable = true;

  __device__ SrcT operator()(SrcT x) {
    return x;
  }
};

// Return an iterator that cast the value to DstT using CastOp.
template <typename DstT, typename Iterator>
__host__ __device__ auto make_cast_iterator(Iterator it) {
  using SrcT = typename cuda::std::iterator_traits<Iterator>::value_type;
  if constexpr (std::is_same_v<SrcT, DstT>) {
    return it;
  } else {
    return thrust::make_transform_iterator(it, CastOp<SrcT, DstT>{});
  }
}

} // namespace mlx::core::cu
