// Copyright © 2025 Apple Inc.

#pragma once

#include <thrust/iterator/iterator_adaptor.h>

namespace mlx::core::cu {

// Always return the value of initial iterator after advancements.
template <typename Iterator>
class repeat_iterator
    : public thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator> {
 public:
  using super_t = thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator>;
  using reference = typename super_t::reference;
  using difference_type = typename super_t::difference_type;

  __host__ __device__ repeat_iterator(Iterator it) : super_t(it), it_(it) {}

 private:
  friend class thrust::iterator_core_access;

  // The dereference is device-only to avoid accidental running in host.
  __device__ typename super_t::reference dereference() const {
    return *it_;
  }

  Iterator it_;
};

} // namespace mlx::core::cu
