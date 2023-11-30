// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"

#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, typename IdxT = int32_t>
struct StridedIterator {
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = IdxT;
  using value_type = T;
  using reference = value_type&;
  using pointer = value_type*;

  // Constructors
  StridedIterator() = default;

  explicit StridedIterator(T* ptr, size_t stride, difference_type offset = 0)
      : ptr_(ptr + offset * stride), stride_(stride) {}

  explicit StridedIterator(array& arr, int axis, difference_type offset = 0)
      : StridedIterator(arr.data<T>(), arr.strides()[axis], offset) {}

  // Accessors
  reference operator*() const {
    return ptr_[0];
  }

  reference operator[](difference_type idx) const {
    return ptr_[idx * stride_];
  }

  // Comparisons
  bool operator==(const StridedIterator& other) const {
    return ptr_ == other.ptr_ && stride_ == other.stride_;
  }

  bool operator!=(const StridedIterator& other) const {
    return ptr_ != other.ptr_;
  }

  bool operator<(const StridedIterator& other) const {
    return ptr_ < other.ptr_;
  }

  bool operator>(const StridedIterator& other) const {
    return ptr_ > other.ptr_;
  }

  bool operator<=(const StridedIterator& other) const {
    return ptr_ <= other.ptr_;
  }

  bool operator>=(const StridedIterator& other) const {
    return ptr_ >= other.ptr_;
  }

  difference_type operator-(const StridedIterator& other) const {
    return (ptr_ - other.ptr_) / stride_;
  }

  // Moving
  StridedIterator& operator++() {
    ptr_ += stride_;
    return *this;
  }

  StridedIterator& operator--() {
    ptr_ -= stride_;
    return *this;
  }

  StridedIterator& operator+=(difference_type diff) {
    ptr_ += diff * stride_;
    return *this;
  }

  StridedIterator& operator-=(difference_type diff) {
    ptr_ -= diff * stride_;
    return *this;
  }

  StridedIterator operator+(difference_type diff) {
    return StridedIterator(ptr_, stride_, diff);
  }

  StridedIterator operator-(difference_type diff) {
    return StridedIterator(ptr_, stride_, -diff);
  }

 private:
  size_t stride_;
  T* ptr_;
};

template <typename T, typename IdxT = uint32_t>
void sort(const array& in, array& out, int axis) {
  // Copy input to output
  CopyType ctype = in.flags().contiguous ? CopyType::Vector : CopyType::General;
  copy(in, out, ctype);

  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in.ndim() : axis;
  size_t n_rows = in.size() / in.shape(axis);

  auto remaining_shape = in.shape();
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = in.strides();
  remaining_strides.erase(remaining_strides.begin() + axis);

  size_t axis_stride = in.strides()[axis];
  int axis_size = in.shape(axis);

  // Perform sorting in place
  for (int i = 0; i < n_rows; i++) {
    size_t loc = elem_to_loc(i, remaining_shape, remaining_strides);
    T* data_ptr = out.data<T>() + loc;

    StridedIterator st(data_ptr, axis_stride, 0);
    StridedIterator ed(data_ptr, axis_stride, axis_size);

    std::stable_sort(st, ed);
  }
}

template <typename T, typename IdxT = uint32_t>
void argsort(const array& in, array& out, int axis) {
  // Allocate output
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in.ndim() : axis;
  size_t n_rows = in.size() / in.shape(axis);

  auto remaining_shape = in.shape();
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = in.strides();
  remaining_strides.erase(remaining_strides.begin() + axis);

  size_t axis_stride = in.strides()[axis];
  int axis_size = in.shape(axis);

  // Perform sorting
  for (int i = 0; i < n_rows; i++) {
    size_t loc = elem_to_loc(i, remaining_shape, remaining_strides);
    const T* data_ptr = in.data<T>() + loc;
    IdxT* idx_ptr = out.data<IdxT>() + loc;

    StridedIterator st_(idx_ptr, axis_stride, 0);
    StridedIterator ed_(idx_ptr, axis_stride, axis_size);

    // Initialize with iota
    std::iota(st_, ed_, IdxT(0));

    // Sort according to vals
    StridedIterator st(idx_ptr, axis_stride, 0);
    StridedIterator ed(idx_ptr, axis_stride, axis_size);

    std::stable_sort(st, ed, [data_ptr, axis_stride](IdxT a, IdxT b) {
      auto v1 = data_ptr[a * axis_stride];
      auto v2 = data_ptr[b * axis_stride];
      return v1 < v2 || (v1 == v2 && a < b);
    });
  }
}

template <typename T, typename IdxT = uint32_t>
void partition(const array& in, array& out, int axis, int kth) {
  // Copy input to output
  CopyType ctype = in.flags().contiguous ? CopyType::Vector : CopyType::General;
  copy(in, out, ctype);

  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in.ndim() : axis;
  size_t n_rows = in.size() / in.shape(axis);

  auto remaining_shape = in.shape();
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = in.strides();
  remaining_strides.erase(remaining_strides.begin() + axis);

  size_t axis_stride = in.strides()[axis];
  int axis_size = in.shape(axis);

  kth = kth < 0 ? kth + axis_size : kth;

  // Perform partition in place
  for (int i = 0; i < n_rows; i++) {
    size_t loc = elem_to_loc(i, remaining_shape, remaining_strides);
    T* data_ptr = out.data<T>() + loc;

    StridedIterator st(data_ptr, axis_stride, 0);
    StridedIterator md(data_ptr, axis_stride, kth);
    StridedIterator ed(data_ptr, axis_stride, axis_size);

    std::nth_element(st, md, ed);
  }
}

template <typename T, typename IdxT = uint32_t>
void argpartition(const array& in, array& out, int axis, int kth) {
  // Allocate output
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in.ndim() : axis;
  size_t n_rows = in.size() / in.shape(axis);

  auto remaining_shape = in.shape();
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = in.strides();
  remaining_strides.erase(remaining_strides.begin() + axis);

  size_t axis_stride = in.strides()[axis];
  int axis_size = in.shape(axis);

  kth = kth < 0 ? kth + axis_size : kth;

  // Perform partition
  for (int i = 0; i < n_rows; i++) {
    size_t loc = elem_to_loc(i, remaining_shape, remaining_strides);
    const T* data_ptr = in.data<T>() + loc;
    IdxT* idx_ptr = out.data<IdxT>() + loc;

    StridedIterator st_(idx_ptr, axis_stride, 0);
    StridedIterator ed_(idx_ptr, axis_stride, axis_size);

    // Initialize with iota
    std::iota(st_, ed_, IdxT(0));

    // Sort according to vals
    StridedIterator st(idx_ptr, axis_stride, 0);
    StridedIterator md(idx_ptr, axis_stride, kth);
    StridedIterator ed(idx_ptr, axis_stride, axis_size);

    std::nth_element(st, md, ed, [data_ptr, axis_stride](IdxT a, IdxT b) {
      auto v1 = data_ptr[a * axis_stride];
      auto v2 = data_ptr[b * axis_stride];
      return v1 < v2 || (v1 == v2 && a < b);
    });
  }
}

} // namespace

void ArgSort::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  switch (in.dtype()) {
    case bool_:
      return argsort<bool>(in, out, axis_);
    case uint8:
      return argsort<uint8_t>(in, out, axis_);
    case uint16:
      return argsort<uint16_t>(in, out, axis_);
    case uint32:
      return argsort<uint32_t>(in, out, axis_);
    case uint64:
      return argsort<uint64_t>(in, out, axis_);
    case int8:
      return argsort<int8_t>(in, out, axis_);
    case int16:
      return argsort<int16_t>(in, out, axis_);
    case int32:
      return argsort<int32_t>(in, out, axis_);
    case int64:
      return argsort<int64_t>(in, out, axis_);
    case float32:
      return argsort<float>(in, out, axis_);
    case float16:
      return argsort<float16_t>(in, out, axis_);
    case bfloat16:
      return argsort<bfloat16_t>(in, out, axis_);
    case complex64:
      return argsort<complex64_t>(in, out, axis_);
  }
}

void Sort::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  switch (in.dtype()) {
    case bool_:
      return sort<bool>(in, out, axis_);
    case uint8:
      return sort<uint8_t>(in, out, axis_);
    case uint16:
      return sort<uint16_t>(in, out, axis_);
    case uint32:
      return sort<uint32_t>(in, out, axis_);
    case uint64:
      return sort<uint64_t>(in, out, axis_);
    case int8:
      return sort<int8_t>(in, out, axis_);
    case int16:
      return sort<int16_t>(in, out, axis_);
    case int32:
      return sort<int32_t>(in, out, axis_);
    case int64:
      return sort<int64_t>(in, out, axis_);
    case float32:
      return sort<float>(in, out, axis_);
    case float16:
      return sort<float16_t>(in, out, axis_);
    case bfloat16:
      return sort<bfloat16_t>(in, out, axis_);
    case complex64:
      return sort<complex64_t>(in, out, axis_);
  }
}

void ArgPartition::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  switch (in.dtype()) {
    case bool_:
      return argpartition<bool>(in, out, axis_, kth_);
    case uint8:
      return argpartition<uint8_t>(in, out, axis_, kth_);
    case uint16:
      return argpartition<uint16_t>(in, out, axis_, kth_);
    case uint32:
      return argpartition<uint32_t>(in, out, axis_, kth_);
    case uint64:
      return argpartition<uint64_t>(in, out, axis_, kth_);
    case int8:
      return argpartition<int8_t>(in, out, axis_, kth_);
    case int16:
      return argpartition<int16_t>(in, out, axis_, kth_);
    case int32:
      return argpartition<int32_t>(in, out, axis_, kth_);
    case int64:
      return argpartition<int64_t>(in, out, axis_, kth_);
    case float32:
      return argpartition<float>(in, out, axis_, kth_);
    case float16:
      return argpartition<float16_t>(in, out, axis_, kth_);
    case bfloat16:
      return argpartition<bfloat16_t>(in, out, axis_, kth_);
    case complex64:
      return argpartition<complex64_t>(in, out, axis_, kth_);
  }
}

void Partition::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  switch (in.dtype()) {
    case bool_:
      return partition<bool>(in, out, axis_, kth_);
    case uint8:
      return partition<uint8_t>(in, out, axis_, kth_);
    case uint16:
      return partition<uint16_t>(in, out, axis_, kth_);
    case uint32:
      return partition<uint32_t>(in, out, axis_, kth_);
    case uint64:
      return partition<uint64_t>(in, out, axis_, kth_);
    case int8:
      return partition<int8_t>(in, out, axis_, kth_);
    case int16:
      return partition<int16_t>(in, out, axis_, kth_);
    case int32:
      return partition<int32_t>(in, out, axis_, kth_);
    case int64:
      return partition<int64_t>(in, out, axis_, kth_);
    case float32:
      return partition<float>(in, out, axis_, kth_);
    case float16:
      return partition<float16_t>(in, out, axis_, kth_);
    case bfloat16:
      return partition<bfloat16_t>(in, out, axis_, kth_);
    case complex64:
      return partition<complex64_t>(in, out, axis_, kth_);
  }
}

} // namespace mlx::core
