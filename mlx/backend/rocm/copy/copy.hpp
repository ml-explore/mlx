// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

namespace mlx::core::rocm {

// Copy function declarations
void copy_contiguous(
    const void* src,
    void* dst,
    size_t size,
    hipStream_t stream);

void copy_general(
    const void* src,
    void* dst,
    const int* src_shape,
    const size_t* src_strides,
    const int* dst_shape,
    const size_t* dst_strides,
    int ndim,
    size_t size,
    size_t dtype_size,
    hipStream_t stream);

void copy_general_dynamic(
    const void* src,
    void* dst,
    const int* src_shape,
    const size_t* src_strides,
    const int* dst_shape,
    const size_t* dst_strides,
    int ndim,
    size_t size,
    size_t dtype_size,
    hipStream_t stream);

void copy_general_input(
    const void* src,
    void* dst,
    const int* src_shape,
    const size_t* src_strides,
    const int* dst_shape,
    const size_t* dst_strides,
    int ndim,
    size_t size,
    size_t dtype_size,
    hipStream_t stream);

// Utility functions for element location calculation
__device__ size_t
elem_to_loc(size_t elem, const int* shape, const size_t* strides, int ndim);

__device__ size_t
loc_to_elem(size_t loc, const int* shape, const size_t* strides, int ndim);

} // namespace mlx::core::rocm