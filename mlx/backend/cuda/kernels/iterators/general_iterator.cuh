// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/utils.cuh"

#include <thrust/iterator/iterator_adaptor.h>
#include <cuda/std/utility>

namespace mlx::core::cu {

// Iterating non-contiguous array.
template <typename Iterator, typename IdxT = int64_t>
class general_iterator
    : public thrust::
          iterator_adaptor<general_iterator<Iterator, IdxT>, Iterator> {
 public:
  using super_t =
      thrust::iterator_adaptor<general_iterator<Iterator, IdxT>, Iterator>;

  using reference = typename super_t::reference;
  using difference_type = typename super_t::difference_type;

  __host__ __device__ general_iterator(
      Iterator it,
      IdxT index,
      int ndim,
      Shape shape,
      Strides strides)
      : super_t(it),
        index_(index),
        ndim_(ndim),
        shape_(cuda::std::move(shape)),
        strides_(cuda::std::move(strides)) {}

  __host__ __device__ IdxT index() const {
    return index_;
  }

  __host__ __device__ const Shape& shape() const {
    return shape_;
  }

  __host__ __device__ const Strides& strides() const {
    return strides_;
  }

 private:
  friend class thrust::iterator_core_access;

  __host__ __device__ bool equal(const general_iterator& other) const {
    return this->base() == other.base() && this->index() == other.index();
  }

  __host__ __device__ void advance(difference_type n) {
    this->index_ += n;
  }

  __host__ __device__ void increment() {
    this->index_ += 1;
  }

  __host__ __device__ void decrement() {
    this->index_ -= 1;
  }

  __host__ __device__ difference_type
  distance_to(const general_iterator& other) const {
    _CCCL_ASSERT(
        this->base() == other.base(),
        "Underlying iterator must point to same base iterator");
    return other.index() - this->index();
  }

  // The dereference is device-only to avoid accidental running in host.
  __device__ typename super_t::reference dereference() const {
    IdxT offset = elem_to_loc(index_, shape_.data(), strides_.data(), ndim_);
    return *(this->base() + offset);
  }

  IdxT index_;
  int ndim_;
  Shape shape_;
  Strides strides_;
};

template <typename IdxT, typename Iterator>
__host__ __device__ auto make_general_iterator(
    Iterator it,
    IdxT index,
    int ndim,
    Shape shape,
    Strides strides) {
  return general_iterator<Iterator, IdxT>(
      it, index, ndim, cuda::std::move(shape), cuda::std::move(strides));
}

template <typename IdxT, typename Iterator>
auto make_general_iterator(
    Iterator it,
    const std::vector<int32_t>& shape,
    const std::vector<int64_t>& strides) {
  return make_general_iterator<IdxT>(
      it, 0, shape.size(), const_param(shape), const_param(strides));
}

template <typename IdxT, typename Iterator>
auto make_general_iterators(
    Iterator it,
    IdxT size,
    const std::vector<int32_t>& shape,
    const std::vector<int64_t>& strides) {
  auto ndim = shape.size();
  auto shape_arg = const_param(shape);
  auto strides_arg = const_param(strides);
  return std::make_pair(
      make_general_iterator<IdxT>(it, 0, ndim, shape_arg, strides_arg),
      make_general_iterator<IdxT>(it, size, ndim, shape_arg, strides_arg));
}

} // namespace mlx::core::cu
