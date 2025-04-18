// Copyright © 2025 Apple Inc.

#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_facade.h>

namespace mlx::core::cu {

// RandomAccessIterator for strided access to array entries.
template <typename Iterator, typename Stride = int64_t>
class strided_iterator
    : public thrust::
          iterator_adaptor<strided_iterator<Iterator, Stride>, Iterator> {
 public:
  using super_t =
      thrust::iterator_adaptor<strided_iterator<Iterator, Stride>, Iterator>;

  using reference = typename super_t::reference;
  using difference_type = typename super_t::difference_type;

  __host__ __device__ strided_iterator(Iterator it, Stride stride)
      : super_t(it), stride_(stride) {}

  __host__ __device__ Stride stride() const {
    return stride_;
  }

 private:
  friend class thrust::iterator_core_access;

  __host__ __device__ bool equal(const strided_iterator& other) const {
    return this->base() == other.base();
  }

  __host__ __device__ void advance(difference_type n) {
    this->base_reference() += n * stride_;
  }

  __host__ __device__ void increment() {
    this->base_reference() += stride_;
  }

  __host__ __device__ void decrement() {
    this->base_reference() -= stride_;
  }

  __host__ __device__ difference_type
  distance_to(const strided_iterator& other) const {
    const difference_type dist = other.base() - this->base();
    _CCCL_ASSERT(
        dist % stride() == 0,
        "Underlying iterator difference must be divisible by the stride");
    return dist / stride();
  }

  Stride stride_;
};

} // namespace mlx::core::cu
