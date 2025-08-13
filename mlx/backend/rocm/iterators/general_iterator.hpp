// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace mlx::core::rocm {

template <typename IdxType>
struct GeneralIterator {
  using difference_type = ptrdiff_t;
  using value_type = IdxType;
  using pointer = IdxType*;
  using reference = IdxType&;
  using iterator_category = std::random_access_iterator_tag;

  const IdxType* base_ptr;
  IdxType offset;
  const int* shape;
  const size_t* strides;
  int ndim;
  size_t size;

  __device__ GeneralIterator(
      const IdxType* base_ptr,
      IdxType offset,
      const int* shape,
      const size_t* strides,
      int ndim,
      size_t size)
      : base_ptr(base_ptr),
        offset(offset),
        shape(shape),
        strides(strides),
        ndim(ndim),
        size(size) {}

  __device__ GeneralIterator operator+(difference_type n) const {
    return GeneralIterator(base_ptr, offset + n, shape, strides, ndim, size);
  }

  __device__ GeneralIterator operator-(difference_type n) const {
    return GeneralIterator(base_ptr, offset - n, shape, strides, ndim, size);
  }

  __device__ difference_type operator-(const GeneralIterator& other) const {
    return offset - other.offset;
  }

  __device__ GeneralIterator& operator+=(difference_type n) {
    offset += n;
    return *this;
  }

  __device__ GeneralIterator& operator-=(difference_type n) {
    offset -= n;
    return *this;
  }

  __device__ GeneralIterator& operator++() {
    ++offset;
    return *this;
  }

  __device__ GeneralIterator operator++(int) {
    GeneralIterator temp = *this;
    ++offset;
    return temp;
  }

  __device__ GeneralIterator& operator--() {
    --offset;
    return *this;
  }

  __device__ GeneralIterator operator--(int) {
    GeneralIterator temp = *this;
    --offset;
    return temp;
  }

  __device__ bool operator==(const GeneralIterator& other) const {
    return offset == other.offset;
  }

  __device__ bool operator!=(const GeneralIterator& other) const {
    return offset != other.offset;
  }

  __device__ bool operator<(const GeneralIterator& other) const {
    return offset < other.offset;
  }

  __device__ bool operator>(const GeneralIterator& other) const {
    return offset > other.offset;
  }

  __device__ bool operator<=(const GeneralIterator& other) const {
    return offset <= other.offset;
  }

  __device__ bool operator>=(const GeneralIterator& other) const {
    return offset >= other.offset;
  }

  __device__ IdxType operator*() const {
    return base_ptr[elem_to_loc(offset, shape, strides, ndim)];
  }

  __device__ IdxType operator[](difference_type n) const {
    return base_ptr[elem_to_loc(offset + n, shape, strides, ndim)];
  }

 private:
  __device__ size_t elem_to_loc(
      size_t elem,
      const int* shape,
      const size_t* strides,
      int ndim) const {
    size_t loc = 0;
    for (int i = ndim - 1; i >= 0; --i) {
      auto q_and_r = div(elem, static_cast<size_t>(shape[i]));
      loc += q_and_r.rem * strides[i];
      elem = q_and_r.quot;
    }
    return loc;
  }

  __device__ div_t div(size_t numer, size_t denom) const {
    div_t result;
    result.quot = numer / denom;
    result.rem = numer % denom;
    return result;
  }
};

template <typename IdxType>
__device__ std::pair<GeneralIterator<IdxType>, GeneralIterator<IdxType>>
make_general_iterators(
    const IdxType* base_ptr,
    size_t size,
    const int* shape,
    const size_t* strides,
    int ndim) {
  auto begin =
      GeneralIterator<IdxType>(base_ptr, 0, shape, strides, ndim, size);
  auto end =
      GeneralIterator<IdxType>(base_ptr, size, shape, strides, ndim, size);
  return std::make_pair(begin, end);
}

} // namespace mlx::core::rocm