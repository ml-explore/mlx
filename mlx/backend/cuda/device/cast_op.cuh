// Copyright © 2025 Apple Inc.

#pragma once

#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// Castings between complex and boolean.
// TODO: Should make a custom complex type.
template <>
struct CastOp<cuComplex, bool> {
  static constexpr bool is_castable = true;

  __device__ bool operator()(cuComplex x) {
    return x.x != 0 && x.y != 0;
  }
};

template <>
struct CastOp<bool, cuComplex> {
  static constexpr bool is_castable = true;

  __device__ cuComplex operator()(bool x) {
    return x ? make_cuFloatComplex(1, 1) : make_cuFloatComplex(0, 0);
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

// Do nothing when no casting is needed.
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

// In CUDA 11 the half types do not define conversions between some types,
// provide fallbacks here.
#if CUDART_VERSION < 12000
template <typename SrcT, typename DstT>
struct CastOp<
    SrcT,
    DstT,
    cuda::std::enable_if_t<
        !cuda::std::is_convertible_v<SrcT, DstT> &&
        !cuda::std::is_same_v<SrcT, cuComplex> &&
        (cuda::std::is_same_v<DstT, __half> ||
         cuda::std::is_same_v<DstT, __nv_bfloat16>)>> {
  static constexpr bool is_castable = true;

  __device__ DstT operator()(SrcT x) {
    return DstT(static_cast<float>(x));
  }
};

template <typename SrcT, typename DstT>
struct CastOp<
    SrcT,
    DstT,
    cuda::std::enable_if_t<
        !cuda::std::is_convertible_v<SrcT, DstT> &&
        !cuda::std::is_same_v<DstT, cuComplex> &&
        !cuda::std::is_same_v<DstT, __half> &&
        !cuda::std::is_same_v<DstT, __nv_bfloat16> &&
        (cuda::std::is_same_v<SrcT, __half> ||
         cuda::std::is_same_v<SrcT, __nv_bfloat16>)>> {
  static constexpr bool is_castable = true;

  __device__ DstT operator()(SrcT x) {
    return DstT(static_cast<float>(x));
  }
};
#endif // CUDART_VERSION < 12000

// Helper to deduce the SrcT.
template <typename DstT, typename SrcT>
inline __host__ __device__ auto cast_to(SrcT x) {
  return CastOp<SrcT, DstT>{}(x);
}

// Return an iterator that cast the value to DstT using CastOp.
template <typename DstT, typename Iterator>
inline __host__ __device__ auto make_cast_iterator(Iterator it) {
  using SrcT = typename cuda::std::iterator_traits<Iterator>::value_type;
  if constexpr (std::is_same_v<SrcT, DstT>) {
    return it;
  } else {
    return thrust::make_transform_iterator(it, CastOp<SrcT, DstT>{});
  }
}

} // namespace mlx::core::cu
