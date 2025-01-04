// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/ops.h"

namespace mlx::core::distributed {

void AllReduce::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  if (inputs[0].is_donatable()) {
    outputs[0].copy_shared_buffer(inputs[0]);
  } else {
    outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));
  }

  switch (reduce_type_) {
    case Sum:
      distributed::detail::all_sum(group(), inputs[0], outputs[0]);
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
      return {{all_sum(inputs[0], group(), stream())}, axes};
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
      return {all_sum(tangents[0], group(), stream())};
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

std::vector<array> AllReduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  return cotangents;
}

void AllGather::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));

  distributed::detail::all_gather(group(), inputs[0], outputs[0]);
}

std::pair<std::vector<array>, std::vector<int>> AllGather::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{all_gather(inputs[0], group(), stream())}, axes};
}

std::vector<array> AllGather::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {all_gather(tangents[0], group(), stream())};
}

std::vector<array> AllGather::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto g = group();
  Shape starts(primals[0].ndim(), 0);
  auto stops = primals[0].shape();
  starts[0] = g.rank() * stops[0];
  stops[0] += starts[0];
  return {slice(cotangents[0], starts, stops)};
}

void Send::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  distributed::detail::send(group(), inputs[0], dst_);
  move_or_copy(inputs[0], outputs[0]);
}

std::pair<std::vector<array>, std::vector<int>> Send::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{send(inputs[0], dst_, group(), stream())}, axes};
}

void Recv::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 0);
  assert(outputs.size() == 1);

  outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));
  distributed::detail::recv(group(), outputs[0], src_);
}

} // namespace mlx::core::distributed
