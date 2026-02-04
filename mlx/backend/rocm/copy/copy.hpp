// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/hip_complex_math.hpp"

#include <hip/hip_runtime.h>
#include <type_traits>

namespace mlx::core {

namespace rocm {

// Type trait for detecting complex types
template <typename T>
struct is_complex : std::false_type {};

template <>
struct is_complex<hipFloatComplex> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Cast operation for copy - general case
template <typename SrcT, typename DstT, typename = void>
struct CastOp {
  static constexpr bool is_castable = std::is_convertible_v<SrcT, DstT>;

  __device__ DstT operator()(SrcT x) {
    return static_cast<DstT>(x);
  }
};

// Castings between complex and boolean
template <>
struct CastOp<hipFloatComplex, bool> {
  static constexpr bool is_castable = true;

  __device__ bool operator()(hipFloatComplex x) {
    return x.x != 0 && x.y != 0;
  }
};

template <>
struct CastOp<bool, hipFloatComplex> {
  static constexpr bool is_castable = true;

  __device__ hipFloatComplex operator()(bool x) {
    return x ? make_hipFloatComplex(1.0f, 1.0f) : make_hipFloatComplex(0.0f, 0.0f);
  }
};

// Converting a complex number to real number discards the imaginary part
template <typename DstT>
struct CastOp<hipFloatComplex, DstT, std::enable_if_t<!is_complex_v<DstT> && !std::is_same_v<DstT, bool>>> {
  static constexpr bool is_castable = true;

  __device__ DstT operator()(hipFloatComplex x) {
    return static_cast<DstT>(x.x);  // x.x is the real part
  }
};

// Allow converting a real number to complex number
template <typename SrcT>
struct CastOp<SrcT, hipFloatComplex, std::enable_if_t<!is_complex_v<SrcT> && !std::is_same_v<SrcT, bool>>> {
  static constexpr bool is_castable = true;

  __device__ hipFloatComplex operator()(SrcT x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

// Do nothing when no casting is needed
template <typename T>
struct CastOp<T, T, void> {
  static constexpr bool is_castable = true;

  __device__ T operator()(T x) {
    return x;
  }
};

// Specializations for half types
template <>
struct CastOp<__half, float> {
  static constexpr bool is_castable = true;
  __device__ float operator()(__half x) {
    return __half2float(x);
  }
};

template <>
struct CastOp<float, __half> {
  static constexpr bool is_castable = true;
  __device__ __half operator()(float x) {
    return __float2half(x);
  }
};

template <>
struct CastOp<hip_bfloat16, float> {
  static constexpr bool is_castable = true;
  __device__ float operator()(hip_bfloat16 x) {
    return static_cast<float>(x);
  }
};

template <>
struct CastOp<float, hip_bfloat16> {
  static constexpr bool is_castable = true;
  __device__ hip_bfloat16 operator()(float x) {
    return hip_bfloat16(x);
  }
};

// Conversions through float for half types
template <typename DstT>
struct CastOp<__half, DstT, std::enable_if_t<!std::is_same_v<DstT, __half> && !std::is_same_v<DstT, float> && !is_complex_v<DstT>>> {
  static constexpr bool is_castable = true;
  __device__ DstT operator()(__half x) {
    return static_cast<DstT>(__half2float(x));
  }
};

template <typename SrcT>
struct CastOp<SrcT, __half, std::enable_if_t<!std::is_same_v<SrcT, __half> && !std::is_same_v<SrcT, float> && !is_complex_v<SrcT>>> {
  static constexpr bool is_castable = true;
  __device__ __half operator()(SrcT x) {
    return __float2half(static_cast<float>(x));
  }
};

template <typename DstT>
struct CastOp<hip_bfloat16, DstT, std::enable_if_t<!std::is_same_v<DstT, hip_bfloat16> && !std::is_same_v<DstT, float> && !is_complex_v<DstT>>> {
  static constexpr bool is_castable = true;
  __device__ DstT operator()(hip_bfloat16 x) {
    return static_cast<DstT>(static_cast<float>(x));
  }
};

template <typename SrcT>
struct CastOp<SrcT, hip_bfloat16, std::enable_if_t<!std::is_same_v<SrcT, hip_bfloat16> && !std::is_same_v<SrcT, float> && !is_complex_v<SrcT>>> {
  static constexpr bool is_castable = true;
  __device__ hip_bfloat16 operator()(SrcT x) {
    return hip_bfloat16(static_cast<float>(x));
  }
};

// Conversion between __half and hip_bfloat16
template <>
struct CastOp<__half, hip_bfloat16> {
  static constexpr bool is_castable = true;
  __device__ hip_bfloat16 operator()(__half x) {
    return hip_bfloat16(__half2float(x));
  }
};

template <>
struct CastOp<hip_bfloat16, __half> {
  static constexpr bool is_castable = true;
  __device__ __half operator()(hip_bfloat16 x) {
    return __float2half(static_cast<float>(x));
  }
};

// Helper to deduce the SrcT
template <typename DstT, typename SrcT>
inline __device__ auto cast_to(SrcT x) {
  return CastOp<SrcT, DstT>{}(x);
}

} // namespace rocm

// Forward declarations
void copy_contiguous(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset);

void copy_general_input(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset,
    const Shape& shape,
    const Strides& strides_in);

void copy_general(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out);

void copy_general_dynamic(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const array& dynamic_offset_in,
    const array& dynamic_offset_out);

} // namespace mlx::core
