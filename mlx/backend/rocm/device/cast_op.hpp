// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

namespace mlx::core::rocm {

// Type trait to check if a type is castable
template <typename From, typename To>
struct is_castable : std::true_type {};

// Cast operation for type conversion
template <typename From, typename To>
struct Cast {
  __device__ To operator()(From x) {
    return static_cast<To>(x);
  }
};

// Same type - no-op
template <typename T>
struct Cast<T, T> {
  __device__ T operator()(T x) {
    return x;
  }
};

// Specializations for half types
template <typename To>
struct Cast<__half, To> {
  __device__ To operator()(__half x) {
    return static_cast<To>(__half2float(x));
  }
};

template <typename From>
struct Cast<From, __half> {
  __device__ __half operator()(From x) {
    return __float2half(static_cast<float>(x));
  }
};

template <>
struct Cast<__half, __half> {
  __device__ __half operator()(__half x) {
    return x;
  }
};

// Specializations for bfloat16 types
template <typename To>
struct Cast<hip_bfloat16, To> {
  __device__ To operator()(hip_bfloat16 x) {
    return static_cast<To>(static_cast<float>(x));
  }
};

template <typename From>
struct Cast<From, hip_bfloat16> {
  __device__ hip_bfloat16 operator()(From x) {
    return hip_bfloat16(static_cast<float>(x));
  }
};

template <>
struct Cast<hip_bfloat16, hip_bfloat16> {
  __device__ hip_bfloat16 operator()(hip_bfloat16 x) {
    return x;
  }
};

// Conversion between half and bfloat16
template <>
struct Cast<__half, hip_bfloat16> {
  __device__ hip_bfloat16 operator()(__half x) {
    return hip_bfloat16(__half2float(x));
  }
};

template <>
struct Cast<hip_bfloat16, __half> {
  __device__ __half operator()(hip_bfloat16 x) {
    return __float2half(static_cast<float>(x));
  }
};

// Complex type conversions
// Complex to bool
template <>
struct Cast<hipFloatComplex, bool> {
  __device__ bool operator()(hipFloatComplex x) {
    return x.x != 0.0f || x.y != 0.0f;
  }
};

// Bool to complex
template <>
struct Cast<bool, hipFloatComplex> {
  __device__ hipFloatComplex operator()(bool x) {
    return make_hipFloatComplex(x ? 1.0f : 0.0f, 0.0f);
  }
};

// Complex to real types (discards imaginary part)
template <>
struct Cast<hipFloatComplex, float> {
  __device__ float operator()(hipFloatComplex x) {
    return x.x;
  }
};

template <>
struct Cast<hipFloatComplex, double> {
  __device__ double operator()(hipFloatComplex x) {
    return static_cast<double>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, int> {
  __device__ int operator()(hipFloatComplex x) {
    return static_cast<int>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, int64_t> {
  __device__ int64_t operator()(hipFloatComplex x) {
    return static_cast<int64_t>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, uint32_t> {
  __device__ uint32_t operator()(hipFloatComplex x) {
    return static_cast<uint32_t>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, uint64_t> {
  __device__ uint64_t operator()(hipFloatComplex x) {
    return static_cast<uint64_t>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, int8_t> {
  __device__ int8_t operator()(hipFloatComplex x) {
    return static_cast<int8_t>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, uint8_t> {
  __device__ uint8_t operator()(hipFloatComplex x) {
    return static_cast<uint8_t>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, int16_t> {
  __device__ int16_t operator()(hipFloatComplex x) {
    return static_cast<int16_t>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, uint16_t> {
  __device__ uint16_t operator()(hipFloatComplex x) {
    return static_cast<uint16_t>(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, __half> {
  __device__ __half operator()(hipFloatComplex x) {
    return __float2half(x.x);
  }
};

template <>
struct Cast<hipFloatComplex, hip_bfloat16> {
  __device__ hip_bfloat16 operator()(hipFloatComplex x) {
    return hip_bfloat16(x.x);
  }
};

// Real types to complex (sets imaginary to 0)
template <>
struct Cast<float, hipFloatComplex> {
  __device__ hipFloatComplex operator()(float x) {
    return make_hipFloatComplex(x, 0.0f);
  }
};

template <>
struct Cast<double, hipFloatComplex> {
  __device__ hipFloatComplex operator()(double x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<int, hipFloatComplex> {
  __device__ hipFloatComplex operator()(int x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<int64_t, hipFloatComplex> {
  __device__ hipFloatComplex operator()(int64_t x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<uint32_t, hipFloatComplex> {
  __device__ hipFloatComplex operator()(uint32_t x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<uint64_t, hipFloatComplex> {
  __device__ hipFloatComplex operator()(uint64_t x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<int8_t, hipFloatComplex> {
  __device__ hipFloatComplex operator()(int8_t x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<uint8_t, hipFloatComplex> {
  __device__ hipFloatComplex operator()(uint8_t x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<int16_t, hipFloatComplex> {
  __device__ hipFloatComplex operator()(int16_t x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<uint16_t, hipFloatComplex> {
  __device__ hipFloatComplex operator()(uint16_t x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

template <>
struct Cast<__half, hipFloatComplex> {
  __device__ hipFloatComplex operator()(__half x) {
    return make_hipFloatComplex(__half2float(x), 0.0f);
  }
};

template <>
struct Cast<hip_bfloat16, hipFloatComplex> {
  __device__ hipFloatComplex operator()(hip_bfloat16 x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

// Complex to complex (identity)
template <>
struct Cast<hipFloatComplex, hipFloatComplex> {
  __device__ hipFloatComplex operator()(hipFloatComplex x) {
    return x;
  }
};

// Helper function for casting (similar to CUDA's cast_to)
template <typename DstT, typename SrcT>
__device__ DstT cast_to(SrcT x) {
  return Cast<SrcT, DstT>{}(x);
}

} // namespace mlx::core::rocm
