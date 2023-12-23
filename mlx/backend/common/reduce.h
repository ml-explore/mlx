// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/utils.h"

namespace mlx::core {

namespace {

enum ReductionOpType {
  // Self-explanatory. Read everything and produce 1 output.
  ContiguousAllReduce,

  // The input is contiguous and the last axis is reduced
  // N1xR1xN2xR2x...xNnxRn
  ContiguousReduce,

  // The input is contiguous and the last axis is not reduced
  // R1xN1xR2xN2x...xRnxNn
  ContiguousStridedReduce,

  // The input is not contiguous but the last axis is and it is reduced so we
  // need to figure out the offsets but we can call the contiguous reduce after
  // that.
  // N3xR1xN1xR4x...xRn
  GeneralContiguousReduce,

  // The input is not contiguous but the last reduction axis and the last axis
  // are so we need to figure out the offset but we can call the strided reduce
  // after that.
  GeneralStridedReduce,

  // The input is not contiguous after the reduction axis and it may contain
  // 0-stride axes or transpositions. We could copy the strides and produce a
  // transposed outcome or we can read the input out of order and write the
  // output in order.
  GeneralReduce
};

// Helper for the ndimensional strided loop
// Should this be in utils?
inline void nd_loop(
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

template <typename T, typename U, typename Op>
struct DefaultStridedReduce {
  Op op;

  DefaultStridedReduce(Op op_) : op(op_) {}

  void operator()(const T* x, U* accumulator, int size, size_t stride) {
    for (int i = 0; i < size; i++) {
      U* moving_accumulator = accumulator;
      for (int j = 0; j < stride; j++) {
        op(moving_accumulator, *x);
        moving_accumulator++;
        x++;
      }
    }
  }
};

template <typename T, typename U, typename Op>
struct DefaultContiguousReduce {
  Op op;

  DefaultContiguousReduce(Op op_) : op(op_) {}

  void operator()(const T* x, U* accumulator, int size) {
    while (size-- > 0) {
      op(accumulator, *x);
      x++;
    }
  }
};

struct ReductionPlan {
  ReductionOpType type;
  std::vector<int> shape;
  std::vector<size_t> strides;

  ReductionPlan(
      ReductionOpType type_,
      std::vector<int> shape_,
      std::vector<size_t> strides_)
      : type(type_), shape(std::move(shape_)), strides(std::move(strides_)) {}
  ReductionPlan(ReductionOpType type_) : type(type_) {}
};

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

template <typename T, typename U, typename OpS, typename OpC, typename Op>
void reduction_op(
    const array& x,
    array& out,
    const std::vector<int>& axes,
    U init,
    OpS ops,
    OpC opc,
    Op op) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  ReductionPlan plan = get_reduction_plan(x, axes);

  if (plan.type == ContiguousAllReduce) {
    U* out_ptr = out.data<U>();
    *out_ptr = init;
    opc(x.data<T>(), out_ptr, x.size());
    return;
  }

  std::vector<int> shape;
  std::vector<size_t> strides;

  if (plan.type == ContiguousReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape[0];
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    for (int i = 0; i < out.size(); i++, out_ptr++, x_ptr += reduction_size) {
      *out_ptr = init;
      opc(x_ptr, out_ptr, reduction_size);
    }
    return;
  }

  if (plan.type == GeneralContiguousReduce || plan.type == ContiguousReduce) {
    int reduction_size = plan.shape.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    // Unrolling the following loop (and implementing it in order for
    // ContiguousReduce) should hold extra performance boost.
    std::tie(shape, strides) = shapes_without_reduction_axes(x, axes);
    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        opc(x_ptr + offset, out_ptr, reduction_size);
      }
    } else {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        nd_loop(
            [&](int extra_offset) {
              opc(x_ptr + offset + extra_offset, out_ptr, reduction_size);
            },
            plan.shape,
            plan.strides);
      }
    }
    return;
  }

  if (plan.type == ContiguousStridedReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    for (int i = 0; i < out.size(); i += reduction_stride) {
      std::fill_n(out_ptr, reduction_stride, init);
      ops(x_ptr, out_ptr, reduction_size, reduction_stride);
      x_ptr += reduction_stride * reduction_size;
      out_ptr += reduction_stride;
    }
    return;
  }

  if (plan.type == GeneralStridedReduce ||
      plan.type == ContiguousStridedReduce) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    std::tie(shape, strides) = shapes_without_reduction_axes(x, axes);
    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        ops(x_ptr + offset, out_ptr, reduction_size, reduction_stride);
        out_ptr += reduction_stride;
      }
    } else {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        nd_loop(
            [&](int extra_offset) {
              ops(x_ptr + offset + extra_offset,
                  out_ptr,
                  reduction_size,
                  reduction_stride);
            },
            plan.shape,
            plan.strides);
        out_ptr += reduction_stride;
      }
    }
    return;
  }

  if (plan.type == GeneralReduce) {
    const T* x_ptr = x.data<T>();
    U* out_ptr = out.data<U>();
    std::tie(shape, strides) = shapes_without_reduction_axes(x, axes);
    for (int i = 0; i < out.size(); i++, out_ptr++) {
      int offset = elem_to_loc(i, shape, strides);
      U val = init;
      nd_loop(
          [&](int extra_offset) { op(&val, *(x_ptr + offset + extra_offset)); },
          plan.shape,
          plan.strides);
      *out_ptr = val;
    }
  }
}

template <typename T, typename U, typename Op>
void reduction_op(
    const array& x,
    array& out,
    const std::vector<int>& axes,
    U init,
    Op op) {
  DefaultStridedReduce<T, U, Op> ops(op);
  DefaultContiguousReduce<T, U, Op> opc(op);
  reduction_op<T, U>(x, out, axes, init, ops, opc, op);
}

} // namespace

} // namespace mlx::core
