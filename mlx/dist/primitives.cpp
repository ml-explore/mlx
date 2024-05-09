// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/dist/ops.h"
#include "mlx/dist/primitives.h"

namespace mlx::core::dist {

void AllReduce::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));

  auto ensure_row_contiguous = [](const array& arr) {
    if (arr.flags().row_contiguous) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      return arr_copy;
    }
  };
  array input = ensure_row_contiguous(inputs[0]);

  switch (reduce_type_) {
    case Sum:
      dist::all_reduce_sum(group(), input, outputs[0]);
      break;
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

std::pair<std::vector<array>, std::vector<int>> AllReduce::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  switch (reduce_type_) {
    case Sum:
      return {{all_reduce_sum(inputs[0], group())}, axes};
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

std::vector<array> AllReduce::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  switch (reduce_type_) {
    case Sum:
      return {all_reduce_sum(tangents[0], group())};
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

std::vector<array> AllReduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  switch (reduce_type_) {
    case Sum:
      return {all_reduce_sum(cotangents[0], group())};
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

} // namespace mlx::core::dist
