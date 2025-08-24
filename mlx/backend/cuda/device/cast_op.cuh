// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/complex.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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
template <typename T>
struct CastOp<complex_t<T>, bool> {
  static constexpr bool is_castable = true;

  __device__ bool operator()(complex_t<T> x) {
    return x.real() != 0 && x.imag() != 0;
  }
};

template <typename T>
struct CastOp<bool, complex_t<T>> {
  static constexpr bool is_castable = true;

  __device__ complex_t<T> operator()(bool x) {
    return x ? complex_t<T>{1, 1} : complex_t<T>{0, 0};
  }
};

// Converting a complex number to real number discards the imaginary part.
template <typename T, typename DstT>
struct CastOp<complex_t<T>, DstT, cuda::std::enable_if_t<!is_complex_v<DstT>>> {
  static constexpr bool is_castable = cuda::std::is_convertible_v<T, DstT>;

  __device__ DstT operator()(complex_t<T> x) {
    static_assert(!is_complex_v<DstT>);
    return static_cast<DstT>(x.real());
  }
};

// Allow converting a real number to complex number.
template <typename SrcT, typename T>
struct CastOp<SrcT, complex_t<T>, cuda::std::enable_if_t<!is_complex_v<SrcT>>> {
  static constexpr bool is_castable = cuda::std::is_convertible_v<SrcT, T>;

  __device__ complex_t<T> operator()(SrcT x) {
    static_assert(!is_complex_v<SrcT>);
    return complex_t<T>{static_cast<T>(x), 0};
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
        !cuda::std::is_convertible_v<SrcT, DstT> && !is_complex_v<SrcT> &&
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
        !cuda::std::is_convertible_v<SrcT, DstT> && !is_complex_v<SrcT> &&
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

} // namespace mlx::core::cu
