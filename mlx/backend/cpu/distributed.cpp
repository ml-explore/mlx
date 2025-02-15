// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/distributed/primitives.h"

namespace mlx::core::distributed {

namespace {

array ensure_row_contiguous(const array& arr, Stream s) {
  if (arr.flags().row_contiguous) {
    return arr;
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy(arr, arr_copy, CopyType::General, s);
    return arr_copy;
  }
}

} // namespace

void AllReduce::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto donate_or_copy = [s = stream()](const array& in, array& out) {
    if (in.flags().row_contiguous) {
      if (in.is_donatable()) {
        out.copy_shared_buffer(in);
      } else {
        out.set_data(allocator::malloc_or_wait(out.nbytes()));
      }
      return in;
    } else {
      array arr_copy(in.shape(), in.dtype(), nullptr, {});
      copy(in, arr_copy, CopyType::General, s);
      out.copy_shared_buffer(arr_copy);
      return arr_copy;
    }
  };

  auto in = donate_or_copy(inputs[0], outputs[0]);
  switch (reduce_type_) {
    case Sum:
      distributed::detail::all_sum(group(), in, outputs[0]);
      break;
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

void AllGather::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto in = ensure_row_contiguous(inputs[0], stream());
  outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));

  distributed::detail::all_gather(group(), in, outputs[0]);
}

void Send::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto in = ensure_row_contiguous(inputs[0], stream());
  distributed::detail::send(group(), in, dst_);
  outputs[0].copy_shared_buffer(inputs[0]);
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
