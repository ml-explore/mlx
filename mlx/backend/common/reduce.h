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
  Shape shape;
  Strides strides;

  ReductionPlan(ReductionOpType type_, Shape shape_, Strides strides_)
      : type(type_), shape(std::move(shape_)), strides(std::move(strides_)) {}
  ReductionPlan(ReductionOpType type_) : type(type_) {}
};

ReductionPlan get_reduction_plan(const array& x, const std::vector<int>& axes);

std::pair<Shape, Strides> shapes_without_reduction_axes(
    const array& x,
    const std::vector<int>& axes);

} // namespace mlx::core
