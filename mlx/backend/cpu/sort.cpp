// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

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
  if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, complex64_t>) {
    if (std::isnan(a))
      return false;
    if (std::isnan(b))
      return true;
  }
  return a < b;
}

// Threshold for switching to linear scan for small axis sizes
constexpr size_t SMALL_LINEAR_THRESHOLD = 32;
// Threshold for using exponential search for large axis sizes
constexpr size_t LARGE_EXP_THRESHOLD = 10000;

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

  StridedIterator operator+(difference_type diff) {
    return StridedIterator(ptr_, stride_, diff);
  }

  StridedIterator operator-(difference_type diff) {
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
      if (std::is_floating_point<T>::value) {
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
      if (std::is_floating_point<T>::value) {
        if (std::isnan(v1))
          return false;
        if (std::isnan(v2))
          return true;
      }

      return v1 < v2 || (v1 == v2 && a < b);
    });
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
      case float64:
        return argsort<double>(in, out, axis_);
      case float16:
        return argsort<float16_t>(in, out, axis_);
      case bfloat16:
        return argsort<bfloat16_t>(in, out, axis_);
      case complex64:
        return argsort<complex64_t>(in, out, axis_);
    }
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
      case float64:
        return argpartition<double>(in, out, axis_, kth_);
      case float16:
        return argpartition<float16_t>(in, out, axis_, kth_);
      case bfloat16:
        return argpartition<bfloat16_t>(in, out, axis_, kth_);
      case complex64:
        return argpartition<complex64_t>(in, out, axis_, kth_);
    }
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
    switch (out.dtype()) {
      case bool_:
        return partition<bool>(out, axis_, kth_);
      case uint8:
        return partition<uint8_t>(out, axis_, kth_);
      case uint16:
        return partition<uint16_t>(out, axis_, kth_);
      case uint32:
        return partition<uint32_t>(out, axis_, kth_);
      case uint64:
        return partition<uint64_t>(out, axis_, kth_);
      case int8:
        return partition<int8_t>(out, axis_, kth_);
      case int16:
        return partition<int16_t>(out, axis_, kth_);
      case int32:
        return partition<int32_t>(out, axis_, kth_);
      case int64:
        return partition<int64_t>(out, axis_, kth_);
      case float32:
        return partition<float>(out, axis_, kth_);
      case float64:
        return partition<double>(out, axis_, kth_);
      case float16:
        return partition<float16_t>(out, axis_, kth_);
      case bfloat16:
        return partition<bfloat16_t>(out, axis_, kth_);
      case complex64:
        return partition<complex64_t>(out, axis_, kth_);
    }
  });
}

namespace {

// Forward declaration
template <typename T, typename IdxT>
void search_sorted(
    const array& a,
    const array& v,
    array& out,
    int axis,
    bool right);

template <typename T, typename IdxT>
void search_sorted_impl(

    const array& a,
    const array& v,
    array& out,
    int axis,
    bool right,
    Stream stream) {
  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  // Get the CPU command encoder and register input and output arrays
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(v);
  encoder.set_output_array(out);

  // Launch the CPU kernel
  encoder.dispatch([a, v, out, axis, right]() mutable {
    // Call the existing search_sorted function inside the dispatched lambda
    search_sorted<T, IdxT>(a, v, out, axis, right);
  });
}

template <typename T, typename IdxT>
void search_sorted(
    const array& a,
    const array& v,
    array& out,
    int axis,
    bool right) {
  auto a_ptr = a.data<T>();
  auto v_ptr = v.data<T>();
  auto out_ptr = out.data<IdxT>();

  auto common_shape = out.shape();
  Strides a_strides = a.strides();
  size_t axis_stride = a_strides[axis];
  size_t axis_size = a.shape(axis);
  a_strides.erase(a_strides.begin() + axis);

  // Contiguous fast-path: when `a` is fully contiguous, the axis has unit
  // stride (innermost dimension), `v` is contiguous and there is a 1:1
  // correspondence between output elements and rows of `a` (no broadcasting).
  // This avoids per-output allocation/copy and strided iterator overhead.
  if (a.flags().contiguous && axis_stride == 1 && v.flags().contiguous &&
      a.size() == out.size() * axis_size && v.size() == out.size()) {
    for (size_t i = 0; i < out.size(); ++i) {
      const T* axis_begin = a_ptr + i * axis_size;
      const T* axis_end = axis_begin + axis_size;

      T val = v_ptr[i];

      IdxT idx;

      // Small arrays: linear scan is often faster due to better cache
      if (axis_size <= SMALL_LINEAR_THRESHOLD) {
        if (right) {
          size_t j = 0;
          for (; j < axis_size; ++j) {
            if (nan_aware_less<T>(val, axis_begin[j]))
              break;
          }
          idx = static_cast<IdxT>(j);
        } else {
          size_t j = 0;
          for (; j < axis_size; ++j) {
            if (!nan_aware_less<T>(axis_begin[j], val))
              break;
          }
          idx = static_cast<IdxT>(j);
        }
      } else if (axis_size > LARGE_EXP_THRESHOLD) {
        // Large arrays: exponential search to reduce binary-search range
        size_t bound = 1;
        if (right) {
          // Expand while value >= a[bound]
          while (bound < axis_size &&
                 !nan_aware_less<T>(val, axis_begin[bound]))
            bound <<= 1;
        } else {
          // Expand while a[bound] < value
          while (bound < axis_size && nan_aware_less<T>(axis_begin[bound], val))
            bound <<= 1;
        }

        size_t left = bound >> 1;
        size_t rightb = std::min(bound, axis_size);

        if (right) {
          auto it = std::upper_bound(
              axis_begin + left, axis_begin + rightb, val, nan_aware_less<T>);
          idx = static_cast<IdxT>(it - axis_begin);
        } else {
          auto it = std::lower_bound(
              axis_begin + left, axis_begin + rightb, val, nan_aware_less<T>);
          idx = static_cast<IdxT>(it - axis_begin);
        }
      } else {
        if (right) {
          auto it =
              std::upper_bound(axis_begin, axis_end, val, nan_aware_less<T>);
          idx = static_cast<IdxT>(it - axis_begin);
        } else {
          auto it =
              std::lower_bound(axis_begin, axis_end, val, nan_aware_less<T>);
          idx = static_cast<IdxT>(it - axis_begin);
        }
      }

      out_ptr[i] = idx;
    }
    return;
  }

  Strides a_broadcast_strides(common_shape.size(), 0);
  Strides v_broadcast_strides(common_shape.size(), 0);

  auto a_shape_no_axis = a.shape();
  a_shape_no_axis.erase(a_shape_no_axis.begin() + axis);

  for (int i = 0; i < common_shape.size(); ++i) {
    int j = common_shape.size() - 1 - i;

    // For v
    int v_dim = v.ndim() - 1 - i;
    if (v_dim >= 0) {
      if (v.shape(v_dim) == 1) {
        v_broadcast_strides[j] = 0;
      } else {
        v_broadcast_strides[j] = v.strides()[v_dim];
      }
    } else {
      v_broadcast_strides[j] = 0;
    }

    // For a
    int a_dim = a_shape_no_axis.size() - 1 - i;
    if (a_dim >= 0) {
      if (a_shape_no_axis[a_dim] == 1) {
        a_broadcast_strides[j] = 0;
      } else {
        a_broadcast_strides[j] = a_strides[a_dim];
      }
    } else {
      a_broadcast_strides[j] = 0;
    }
  }

  ContiguousIterator a_it(
      common_shape, a_broadcast_strides, common_shape.size());
  ContiguousIterator v_it(
      common_shape, v_broadcast_strides, common_shape.size());

  for (size_t i = 0; i < out.size(); ++i) {
    T val = v_ptr[v_it.loc];
    size_t a_offset = a_it.loc;

    const T* base_ptr = a_ptr + a_offset;

    // Use strided iterators directly to avoid per-output heap allocation and
    // copying of the axis slice.
    StridedIterator<const T> axis_begin(
        base_ptr, static_cast<int64_t>(axis_stride), 0);
    StridedIterator<const T> axis_end(
        base_ptr, static_cast<int64_t>(axis_stride), axis_size);

    IdxT idx;

    // Small arrays: use linear scan over strided axis
    if (axis_size <= SMALL_LINEAR_THRESHOLD) {
      if (right) {
        size_t j = 0;
        for (; j < axis_size; ++j) {
          const T& aelem = axis_begin[j];
          if (nan_aware_less<T>(val, aelem))
            break;
        }
        idx = static_cast<IdxT>(j);
      } else {
        size_t j = 0;
        for (; j < axis_size; ++j) {
          const T& aelem = axis_begin[j];
          if (!nan_aware_less<T>(aelem, val))
            break;
        }
        idx = static_cast<IdxT>(j);
      }
    } else if (axis_size > LARGE_EXP_THRESHOLD) {
      // Large arrays: exponential search over strided axis
      size_t bound = 1;
      if (right) {
        while (bound < axis_size && !nan_aware_less<T>(val, axis_begin[bound]))
          bound <<= 1;
      } else {
        while (bound < axis_size && nan_aware_less<T>(axis_begin[bound], val))
          bound <<= 1;
      }

      size_t left = bound >> 1;
      size_t rightb = std::min(bound, axis_size);

      if (right) {
        auto it = std::upper_bound(
            axis_begin + left, axis_begin + rightb, val, nan_aware_less<T>);
        idx = static_cast<IdxT>(it - axis_begin);
      } else {
        auto it = std::lower_bound(
            axis_begin + left, axis_begin + rightb, val, nan_aware_less<T>);
        idx = static_cast<IdxT>(it - axis_begin);
      }
    } else {
      if (right) {
        auto it =
            std::upper_bound(axis_begin, axis_end, val, nan_aware_less<T>);
        idx = static_cast<IdxT>(it - axis_begin);
      } else {
        auto it =
            std::lower_bound(axis_begin, axis_end, val, nan_aware_less<T>);
        idx = static_cast<IdxT>(it - axis_begin);
      }
    }
    out_ptr[i] = idx;

    a_it.step();
    v_it.step();
  }
}

} // namespace

void SearchSorted::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& a = inputs[0];
  auto& v = inputs[1];
  auto& out = outputs[0];

  if (out.size() == 0) {
    return;
  }

  int ax = axis_;
  if (ax < 0) {
    ax += a.ndim();
  }

  switch (a.dtype()) {
    case bool_:
      search_sorted_impl<bool, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case uint8:
      search_sorted_impl<uint8_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case uint16:
      search_sorted_impl<uint16_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case uint32:
      search_sorted_impl<uint32_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case uint64:
      search_sorted_impl<uint64_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case int8:
      search_sorted_impl<int8_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case int16:
      search_sorted_impl<int16_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case int32:
      search_sorted_impl<int32_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case int64:
      search_sorted_impl<int64_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case float16:
      search_sorted_impl<float16_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case bfloat16:
      search_sorted_impl<bfloat16_t, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case float32:
      search_sorted_impl<float, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case float64:
      search_sorted_impl<double, uint32_t>(a, v, out, ax, right_, stream());
      break;
    case complex64:
      search_sorted_impl<complex64_t, uint32_t>(
          a, v, out, ax, right_, stream());
      break;
  }
}

} // namespace mlx::core
