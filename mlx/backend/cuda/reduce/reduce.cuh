// Copyright © 2025 Apple Inc.

#include <type_traits>

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/cuda/device/cucomplex_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/reduce/reduce_ops.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename F>
void dispatch_reduce_ndim(int ndim, F&& f) {
  if (ndim == 1) {
    f(std::integral_constant<int, 1>{});
  } else if (ndim == 2) {
    f(std::integral_constant<int, 2>{});
  } else {
    f(std::integral_constant<int, 5>{});
  }
}

template <typename F>
void dispatch_reduce_ops(Reduce::ReduceType reduce_type, F&& f) {
  if (reduce_type == Reduce::ReduceType::And) {
    f(type_identity<cu::And>{});
  } else if (reduce_type == Reduce::ReduceType::Or) {
    f(type_identity<cu::Or>{});
  } else if (reduce_type == Reduce::ReduceType::Sum) {
    f(type_identity<cu::Sum>{});
  } else if (reduce_type == Reduce::ReduceType::Prod) {
    f(type_identity<cu::Prod>{});
  } else if (reduce_type == Reduce::ReduceType::Max) {
    f(type_identity<cu::Max>{});
  } else if (reduce_type == Reduce::ReduceType::Min) {
    f(type_identity<cu::Min>{});
  } else {
    throw std::invalid_argument("Unknown reduce type.");
  }
}

void all_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type);

void row_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

void col_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

void init_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type);

} // namespace mlx::core
