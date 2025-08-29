// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace mlx::core::rocm {

template <typename T>
struct StridedIterator {
  using difference_type = ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::random_access_iterator_tag;

  T* ptr;
  size_t stride;

  __device__ StridedIterator(T* ptr, size_t stride)
      : ptr(ptr), stride(stride) {}

  __device__ StridedIterator operator+(difference_type n) const {
    return StridedIterator(ptr + n * stride, stride);
  }

  __device__ StridedIterator operator-(difference_type n) const {
    return StridedIterator(ptr - n * stride, stride);
  }

  __device__ difference_type operator-(const StridedIterator& other) const {
    return (ptr - other.ptr) / stride;
  }

  __device__ StridedIterator& operator+=(difference_type n) {
    ptr += n * stride;
    return *this;
  }

  __device__ StridedIterator& operator-=(difference_type n) {
    ptr -= n * stride;
    return *this;
  }

  __device__ StridedIterator& operator++() {
    ptr += stride;
    return *this;
  }

  __device__ StridedIterator operator++(int) {
    StridedIterator temp = *this;
    ptr += stride;
    return temp;
  }

  __device__ StridedIterator& operator--() {
    ptr -= stride;
    return *this;
  }

  __device__ StridedIterator operator--(int) {
    StridedIterator temp = *this;
    ptr -= stride;
    return temp;
  }

  __device__ bool operator==(const StridedIterator& other) const {
    return ptr == other.ptr;
  }

  __device__ bool operator!=(const StridedIterator& other) const {
    return ptr != other.ptr;
  }

  __device__ bool operator<(const StridedIterator& other) const {
    return ptr < other.ptr;
  }

  __device__ bool operator>(const StridedIterator& other) const {
    return ptr > other.ptr;
  }

  __device__ bool operator<=(const StridedIterator& other) const {
    return ptr <= other.ptr;
  }

  __device__ bool operator>=(const StridedIterator& other) const {
    return ptr >= other.ptr;
  }

  __device__ T& operator*() const {
    return *ptr;
  }

  __device__ T& operator[](difference_type n) const {
    return *(ptr + n * stride);
  }
};

template <typename T>
__device__ StridedIterator<T> make_strided_iterator(T* ptr, size_t stride) {
  return StridedIterator<T>(ptr, stride);
}

} // namespace mlx::core::rocm