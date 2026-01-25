// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Cast operation for copy
template <typename Out, typename In>
__device__ Out cast_to(In x) {
  return static_cast<Out>(x);
}

// Specializations for half types
template <>
__device__ inline float cast_to<float, __half>(__half x) {
  return __half2float(x);
}

template <>
__device__ inline __half cast_to<__half, float>(float x) {
  return __float2half(x);
}

template <>
__device__ inline float cast_to<float, hip_bfloat16>(hip_bfloat16 x) {
  return static_cast<float>(x);
}

template <>
__device__ inline hip_bfloat16 cast_to<hip_bfloat16, float>(float x) {
  return hip_bfloat16(x);
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

} // namespace mlx::core
