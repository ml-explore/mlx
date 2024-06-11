// Copyright © 2023 Apple Inc.

#include <cassert>
#include <functional>
#include <limits>

#include "mlx/backend/common/reduce.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename U>
struct Limits {
  static const U max;
  static const U min;
};

#define instantiate_default_limit(type)                           \
  template <>                                                     \
  struct Limits<type> {                                           \
    static constexpr type max = std::numeric_limits<type>::max(); \
    static constexpr type min = std::numeric_limits<type>::min(); \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type) \
  template <>                         \
  struct Limits<type> {               \
    static const type max;            \
    static const type min;            \
  };

instantiate_float_limit(float16_t);
instantiate_float_limit(bfloat16_t);
instantiate_float_limit(float);
instantiate_float_limit(complex64_t);

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

const float Limits<float>::max = std::numeric_limits<float>::infinity();
const float Limits<float>::min = -std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::max =
    std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::min =
    -std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::max = std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::min =
    -std::numeric_limits<float>::infinity();
const complex64_t Limits<complex64_t>::max =
    std::numeric_limits<float>::infinity();
const complex64_t Limits<complex64_t>::min =
    -std::numeric_limits<float>::infinity();

struct AndReduce {
  template <typename T>
  void operator()(bool* a, T b) {
    (*a) &= (b != 0);
  }

  void operator()(bool* y, bool x) {
    (*y) &= x;
  }
};

struct OrReduce {
  template <typename T>
  void operator()(bool* a, T b) {
    (*a) |= (b != 0);
  }

  void operator()(bool* y, bool x) {
    (*y) |= x;
  }
};

template <typename InT>
void reduce_dispatch_out(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  switch (rtype) {
    case Reduce::And: {
      reduction_op<InT, bool>(in, out, axes, true, AndReduce());
      break;
    }
    case Reduce::Or: {
      reduction_op<InT, bool>(in, out, axes, false, OrReduce());
      break;
    }
    case Reduce::Sum: {
      auto op = [](auto y, auto x) { (*y) = (*y) + x; };
      if (out.dtype() == int32) {
        // special case since the input type can be bool
        reduction_op<InT, int32_t>(in, out, axes, 0, op);
      } else {
        reduction_op<InT, InT>(in, out, axes, 0, op);
      }
      break;
    }
    case Reduce::Prod: {
      auto op = [](auto y, auto x) { (*y) *= x; };
      reduction_op<InT, InT>(in, out, axes, 1, op);
      break;
    }
    case Reduce::Max: {
      auto op = [](auto y, auto x) { (*y) = (*y > x) ? *y : x; };
      auto init = Limits<InT>::min;
      reduction_op<InT, InT>(in, out, axes, init, op);
      break;
    }
    case Reduce::Min: {
      auto op = [](auto y, auto x) { (*y) = (*y < x) ? *y : x; };
      auto init = Limits<InT>::max;
      reduction_op<InT, InT>(in, out, axes, init, op);
      break;
    }
  }
}

} // namespace

void nd_loop(
    std::function<void(int)> callback,
    const std::vector<int>& shape,
    const std::vector<size_t>& strides) {
  std::function<void(int, int)> loop_inner;
  loop_inner = [&](int dim, int offset) {
    if (dim < shape.size() - 1) {
      int size = shape[dim];
      size_t stride = strides[dim];
      for (int i = 0; i < size; i++) {
        loop_inner(dim + 1, offset + i * stride);
      }
    } else {
      int size = shape[dim];
      size_t stride = strides[dim];
      for (int i = 0; i < size; i++) {
        callback(offset + i * stride);
      }
    }
  };
  loop_inner(0, 0);
}

std::pair<std::vector<int>, std::vector<size_t>> shapes_without_reduction_axes(
    const array& x,
    const std::vector<int>& axes) {
  std::vector<int> shape = x.shape();
  std::vector<size_t> strides = x.strides();

  for (int i = axes.size() - 1; i >= 0; i--) {
    int a = axes[i];
    shape.erase(shape.begin() + a);
    strides.erase(strides.begin() + a);
  }

  return std::make_pair(shape, strides);
}

ReductionPlan get_reduction_plan(const array& x, const std::vector<int> axes) {
  // The data is all there and we are reducing over everything
  if (x.size() == x.data_size() && axes.size() == x.ndim() &&
      x.flags().contiguous) {
    return ContiguousAllReduce;
  }

  // Row contiguous input so the output is row contiguous
  if (x.flags().row_contiguous) {
    // Merge consecutive axes
    std::vector<int> shape = {x.shape(axes[0])};
    std::vector<size_t> strides = {x.strides()[axes[0]]};
    for (int i = 1; i < axes.size(); i++) {
      if (axes[i] - 1 == axes[i - 1]) {
        shape.back() *= x.shape(axes[i]);
        strides.back() = x.strides()[axes[i]];
      } else {
        shape.push_back(x.shape(axes[i]));
        strides.push_back(x.strides()[axes[i]]);
      }
    }

    if (strides.back() == 1) {
      return ReductionPlan(ContiguousReduce, shape, strides);
    } else if (strides.back() > 1) {
      return ReductionPlan(ContiguousStridedReduce, shape, strides);
    }
  }

  // Let's check if we can optimize our access patterns
  //
  // 1. We have a reduction axis with stride 1. Simply call
  //    GeneralContiguousReduce and be done with it.
  // 2. We have transpositions and we are not reducing over the axis with
  //    stride 1. However, we are reducing over an axis where everything is
  //    contiguous in memory to the right of that axis. We can call strided
  //    reduce and be done with it.
  // 2. We have weird transpositions and expands. Copy the strides to the
  //    output, then call strided reduce.

  // Sort reduction axes by stride in order to merge them and figure out if we
  // have a contiguous reduction.
  std::vector<std::pair<int, size_t>> reductions;
  for (auto a : axes) {
    reductions.push_back(std::make_pair(x.shape(a), x.strides()[a]));
  }
  std::sort(reductions.begin(), reductions.end(), [](auto a, auto b) {
    return a.second > b.second;
  });
  // Extract the two smallest and try to merge them in case the contiguous
  // reduction can be bigger than just the last axis.
  for (int i = reductions.size() - 1; i >= 1; i--) {
    auto a = reductions[i];
    auto b = reductions[i - 1];

    // b.stride = a.shape * a.stride then a and b are contiguous
    if (b.second == a.first * a.second) {
      reductions.erase(reductions.begin() + i);
      reductions[i - 1] = std::make_pair(a.first * b.first, a.second);
    }
  }

  std::vector<int> shape;
  std::vector<size_t> strides;
  for (auto r : reductions) {
    shape.push_back(r.first);
    strides.push_back(r.second);
  }

  // We can call the contiguous reduction op for every weird way the input is
  // structured in the rest of the axes.
  if (strides.back() == 1) {
    return ReductionPlan(GeneralContiguousReduce, shape, strides);
  }

  // Delegate to the general strided reduction op if the axes after
  // strides.back() are contiguous.
  if (strides.back() > 1) {
    int size = 1;
    for (int i = x.ndim() - 1; i >= 0; i--) {
      if (axes.back() == i) {
        continue;
      }
      if (x.strides()[i] != size) {
        break;
      }
      size *= x.shape(i);
    }
    if (size >= strides.back()) {
      return ReductionPlan(GeneralStridedReduce, shape, strides);
    }
  }

  return ReductionPlan(GeneralReduce, shape, strides);
}

void Reduce::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  switch (in.dtype()) {
    case bool_:
      reduce_dispatch_out<bool>(in, out, reduce_type_, axes_);
      break;
    case uint8:
      reduce_dispatch_out<uint8_t>(in, out, reduce_type_, axes_);
      break;
    case uint16:
      reduce_dispatch_out<uint16_t>(in, out, reduce_type_, axes_);
      break;
    case uint32:
      reduce_dispatch_out<uint32_t>(in, out, reduce_type_, axes_);
      break;
    case uint64:
      reduce_dispatch_out<uint64_t>(in, out, reduce_type_, axes_);
      break;
    case int8:
      reduce_dispatch_out<uint8_t>(in, out, reduce_type_, axes_);
      break;
    case int16:
      reduce_dispatch_out<uint16_t>(in, out, reduce_type_, axes_);
      break;
    case int32:
      reduce_dispatch_out<int32_t>(in, out, reduce_type_, axes_);
      break;
    case int64:
      reduce_dispatch_out<int64_t>(in, out, reduce_type_, axes_);
      break;
    case float16:
      reduce_dispatch_out<float16_t>(in, out, reduce_type_, axes_);
      break;
    case float32:
      reduce_dispatch_out<float>(in, out, reduce_type_, axes_);
      break;
    case bfloat16:
      reduce_dispatch_out<bfloat16_t>(in, out, reduce_type_, axes_);
      break;
    case complex64:
      reduce_dispatch_out<complex64_t>(in, out, reduce_type_, axes_);
      break;
  }
}

} // namespace mlx::core
