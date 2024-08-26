// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/utils.h"

namespace mlx::core {

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

ReductionPlan get_reduction_plan(const array& x, const std::vector<int>& axes);

// Helper for the ndimensional strided loop
// Should this be in utils?
void nd_loop(
    std::function<void(int)> callback,
    const std::vector<int>& shape,
    const std::vector<size_t>& strides);

std::pair<std::vector<int>, std::vector<size_t>> shapes_without_reduction_axes(
    const array& x,
    const std::vector<int>& axes);

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

} // namespace mlx::core
