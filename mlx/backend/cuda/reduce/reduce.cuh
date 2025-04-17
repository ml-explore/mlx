// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/kernels/cucomplex_math.cuh"
#include "mlx/backend/cuda/kernels/reduce_ops.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

// Dispatch dynamic ndim to constexpr.
#define MORE_THAN_TWO 5
#define MLX_SWITCH_NDIM(ndim, NDIM, ...)     \
  if (ndim == 1) {                           \
    constexpr uint32_t NDIM = 1;             \
    __VA_ARGS__;                             \
  } else if (ndim == 2) {                    \
    constexpr uint32_t NDIM = 2;             \
    __VA_ARGS__;                             \
  } else {                                   \
    constexpr uint32_t NDIM = MORE_THAN_TWO; \
    __VA_ARGS__;                             \
  }

// Dispatch reduce ops to constexpr.
#define MLX_SWITCH_REDUCE_OPS(REDUCE, OP, ...)           \
  if (REDUCE == Reduce::ReduceType::And) {               \
    using OP = cu::And;                                  \
    __VA_ARGS__;                                         \
  } else if (REDUCE == Reduce::ReduceType::Or) {         \
    using OP = cu::Or;                                   \
    __VA_ARGS__;                                         \
  } else if (REDUCE == Reduce::ReduceType::Sum) {        \
    using OP = cu::Sum;                                  \
    __VA_ARGS__;                                         \
  } else if (REDUCE == Reduce::ReduceType::Prod) {       \
    using OP = cu::Prod;                                 \
    __VA_ARGS__;                                         \
  } else if (REDUCE == Reduce::ReduceType::Max) {        \
    using OP = cu::Max;                                  \
    __VA_ARGS__;                                         \
  } else if (REDUCE == Reduce::ReduceType::Min) {        \
    using OP = cu::Min;                                  \
    __VA_ARGS__;                                         \
  } else {                                               \
    throw std::invalid_argument("Unknown reduce type."); \
  }

void segmented_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

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

} // namespace mlx::core
