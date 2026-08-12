// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// NaN-aware comparator that places NaNs at the end
template <typename T>
bool nan_aware_less(T a, T b) {
  if constexpr (is_floating_point_v<T> || std::is_same_v<T, complex64_t>) {
    if (std::isnan(a))
      return false;
    if (std::isnan(b))
      return true;
  }
  return a < b;
}

template <typename T>
struct StridedIterator {
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = int32_t;
  using value_type = T;
  using reference = value_type&;
  using pointer = value_type*;

  // Constructors
  StridedIterator() = default;

  explicit StridedIterator(T* ptr, int64_t stride, difference_type offset = 0)
      : stride_(stride), ptr_(ptr + offset * stride) {}

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

  StridedIterator operator+(difference_type diff) const {
    return StridedIterator(ptr_, stride_, diff);
  }

  StridedIterator operator-(difference_type diff) const {
    return StridedIterator(ptr_, stride_, -diff);
  }

 private:
  int64_t stride_;
  T* ptr_;
};

template <typename T>
void sort(array& out, int axis) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + out.ndim() : axis;
  size_t in_size = out.size();
  size_t n_rows = in_size / out.shape(axis);

  auto remaining_shape = out.shape();
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = out.strides();
  remaining_strides.erase(remaining_strides.begin() + axis);

  auto axis_stride = out.strides()[axis];
  auto axis_size = out.shape(axis);

  // Perform sorting in place
  ContiguousIterator src_it(
      remaining_shape, remaining_strides, remaining_shape.size());
  auto out_ptr = out.data<T>();
  for (int i = 0; i < n_rows; i++) {
    T* data_ptr = out_ptr + src_it.loc;

    StridedIterator st(data_ptr, axis_stride, 0);
    StridedIterator ed(data_ptr, axis_stride, axis_size);

    std::stable_sort(st, ed, nan_aware_less<T>);
    src_it.step();
  }
}

template <typename T, typename IdxT = uint32_t>
void argsort(const array& in, array& out, int axis) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in.ndim() : axis;
  size_t n_rows = in.size() / in.shape(axis);

  auto in_remaining_shape = in.shape();
  in_remaining_shape.erase(in_remaining_shape.begin() + axis);

  auto in_remaining_strides = in.strides();
  in_remaining_strides.erase(in_remaining_strides.begin() + axis);

  auto out_remaining_shape = out.shape();
  out_remaining_shape.erase(out_remaining_shape.begin() + axis);

  auto out_remaining_strides = out.strides();
  out_remaining_strides.erase(out_remaining_strides.begin() + axis);

  auto in_stride = in.strides()[axis];
  auto out_stride = out.strides()[axis];
  auto axis_size = in.shape(axis);

  // Perform sorting
  ContiguousIterator in_it(
      in_remaining_shape, in_remaining_strides, in_remaining_shape.size());
  ContiguousIterator out_it(
      out_remaining_shape, out_remaining_strides, out_remaining_shape.size());
  auto in_ptr = in.data<T>();
  auto out_ptr = out.data<IdxT>();
  for (int i = 0; i < n_rows; i++) {
    const T* data_ptr = in_ptr + in_it.loc;
    IdxT* idx_ptr = out_ptr + out_it.loc;

    in_it.step();
    out_it.step();

    StridedIterator st_(idx_ptr, out_stride, 0);
    StridedIterator ed_(idx_ptr, out_stride, axis_size);

    // Initialize with iota
    std::iota(st_, ed_, IdxT(0));

    // Sort according to vals
    StridedIterator st(idx_ptr, out_stride, 0);
    StridedIterator ed(idx_ptr, out_stride, axis_size);

    std::stable_sort(st, ed, [data_ptr, in_stride](IdxT a, IdxT b) {
      auto v1 = data_ptr[a * in_stride];
      auto v2 = data_ptr[b * in_stride];

      // Handle NaNs (place them at the end)
      if constexpr (is_floating_point_v<T>) {
        if (std::isnan(v1))
          return false;
        if (std::isnan(v2))
          return true;
      }

      return v1 < v2 || (v1 == v2 && a < b);
    });
  }
}

template <typename T>
void partition(array& out, int axis, int kth) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + out.ndim() : axis;
  size_t in_size = out.size();
  size_t n_rows = in_size / out.shape(axis);

  auto remaining_shape = out.shape();
  remaining_shape.erase(remaining_shape.begin() + axis);

  auto remaining_strides = out.strides();
  remaining_strides.erase(remaining_strides.begin() + axis);

  auto axis_stride = out.strides()[axis];
  int axis_size = out.shape(axis);

  kth = kth < 0 ? kth + axis_size : kth;

  // Perform partition in place
  ContiguousIterator src_it(
      remaining_shape, remaining_strides, remaining_shape.size());
  auto out_ptr = out.data<T>();
  for (int i = 0; i < n_rows; i++) {
    T* data_ptr = out_ptr + src_it.loc;
    src_it.step();

    StridedIterator st(data_ptr, axis_stride, 0);
    StridedIterator md(data_ptr, axis_stride, kth);
    StridedIterator ed(data_ptr, axis_stride, axis_size);

    std::nth_element(st, md, ed, nan_aware_less<T>);
  }
}

template <typename T, typename IdxT = uint32_t>
void argpartition(const array& in, array& out, int axis, int kth) {
  // Get axis, shape and stride info
  axis = axis < 0 ? axis + in.ndim() : axis;
  size_t n_rows = in.size() / in.shape(axis);

  auto in_remaining_shape = in.shape();
  in_remaining_shape.erase(in_remaining_shape.begin() + axis);

  auto in_remaining_strides = in.strides();
  in_remaining_strides.erase(in_remaining_strides.begin() + axis);

  auto out_remaining_shape = out.shape();
  out_remaining_shape.erase(out_remaining_shape.begin() + axis);

  auto out_remaining_strides = out.strides();
  out_remaining_strides.erase(out_remaining_strides.begin() + axis);

  auto in_stride = in.strides()[axis];
  auto out_stride = out.strides()[axis];
  auto axis_size = in.shape(axis);

  kth = kth < 0 ? kth + axis_size : kth;

  // Perform partition
  ContiguousIterator in_it(
      in_remaining_shape, in_remaining_strides, in_remaining_shape.size());
  ContiguousIterator out_it(
      out_remaining_shape, out_remaining_strides, out_remaining_shape.size());

  auto in_ptr = in.data<T>();
  auto out_ptr = out.data<IdxT>();

  for (int i = 0; i < n_rows; i++) {
    const T* data_ptr = in_ptr + in_it.loc;
    IdxT* idx_ptr = out_ptr + out_it.loc;
    in_it.step();
    out_it.step();

    StridedIterator st_(idx_ptr, out_stride, 0);
    StridedIterator ed_(idx_ptr, out_stride, axis_size);

    // Initialize with iota
    std::iota(st_, ed_, IdxT(0));

    // Sort according to vals
    StridedIterator st(idx_ptr, out_stride, 0);
    StridedIterator md(idx_ptr, out_stride, kth);
    StridedIterator ed(idx_ptr, out_stride, axis_size);

    std::nth_element(st, md, ed, [data_ptr, in_stride](IdxT a, IdxT b) {
      auto v1 = data_ptr[a * in_stride];
      auto v2 = data_ptr[b * in_stride];

      // Handle NaNs (place them at the end)
      if constexpr (is_floating_point_v<T>) {
        if (std::isnan(v1))
          return false;
        if (std::isnan(v2))
          return true;
      }

      return v1 < v2 || (v1 == v2 && a < b);
    });
  }
}

template <typename T, bool Right>
void searchsorted(const array& a, const array& v, array& out) {
  auto n = static_cast<uint32_t>(a.size());
  auto a_stride = a.strides()[0]; // sequence is 1D
  const T* a_ptr = a.data<T>();
  const T* v_ptr = v.data<T>();
  uint32_t* out_ptr = out.data<uint32_t>();

  auto bound = [a_ptr, a_stride, n](T x) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      T m = a_ptr[static_cast<int64_t>(mid) * a_stride];
      bool below = Right ? !nan_aware_less(x, m) : nan_aware_less(m, x);
      if (below) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  };

  if (v.flags().row_contiguous) {
    for (size_t i = 0; i < v.size(); ++i) {
      out_ptr[i] = bound(v_ptr[i]);
    }
  } else {
    ContiguousIterator it(v);
    for (size_t i = 0; i < v.size(); ++i) {
      out_ptr[i] = bound(v_ptr[it.loc]);
      it.step();
    }
  }
}

} // namespace

void ArgSort::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_input_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    axis_ = axis_]() mutable {
    dispatch_all_types(in.dtype(), [&](auto type_tag) {
      argsort<MLX_GET_TYPE(type_tag)>(in, out, axis_);
    });
  });
}

void Sort::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  int axis = axis_;
  if (axis < 0) {
    axis += in.ndim();
  }

  // Copy input to output
  CopyType ctype = (in.flags().contiguous && in.strides()[axis] != 0)
      ? CopyType::Vector
      : CopyType::General;
  copy_cpu(in, out, ctype, stream());

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_output_array(out);
  encoder.dispatch([out = array::unsafe_weak_copy(out), axis]() mutable {
    dispatch_all_types(out.dtype(), [&](auto type_tag) {
      sort<MLX_GET_TYPE(type_tag)>(out, axis);
    });
  });
}

void ArgPartition::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_input_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    axis_ = axis_,
                    kth_ = kth_]() mutable {
    dispatch_all_types(in.dtype(), [&](auto type_tag) {
      argpartition<MLX_GET_TYPE(type_tag)>(in, out, axis_, kth_);
    });
  });
}

void Partition::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Copy input to output
  CopyType ctype = (in.flags().contiguous && in.strides()[axis_] != 0)
      ? CopyType::Vector
      : CopyType::General;
  copy_cpu(in, out, ctype, stream());

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_output_array(out);
  encoder.dispatch([out = array::unsafe_weak_copy(out),
                    axis_ = axis_,
                    kth_ = kth_]() mutable {
    dispatch_all_types(out.dtype(), [&](auto type_tag) {
      partition<MLX_GET_TYPE(type_tag)>(out, axis_, kth_);
    });
  });
}

void SearchSorted::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& v = inputs[1];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(a);
  encoder.set_input_array(v);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    v = array::unsafe_weak_copy(v),
                    out = array::unsafe_weak_copy(out),
                    right = right_]() mutable {
    dispatch_all_types(a.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      if (right) {
        searchsorted<T, true>(a, v, out);
      } else {
        searchsorted<T, false>(a, v, out);
      }
    });
  });
}

} // namespace mlx::core
