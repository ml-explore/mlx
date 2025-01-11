// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/event.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::distributed {

void signal_and_wait(const array& in, const array& out) {
  if (in.event().valid()) {
    encode_signal(in.event());
  }
  encode_wait(out.event());
}

void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& in = inputs[0];
  auto& out = outputs[0];
  if (in.is_donatable()) {
    out.move_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }

  auto task = [in = in,
               out = out,
               reduce_type = reduce_type_,
               group = group()]() mutable {
    if (in.event().valid()) {
      in.event().wait();
    }
    switch (reduce_type) {
      case Sum:
        distributed::detail::all_sum(
            group, in.data_shared_ptr() == nullptr ? out : in, out);
        break;
      default:
        throw std::runtime_error("Only all reduce sum is supported for now");
    }
    out.event().signal();
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));

  signal_and_wait(in, out);
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& out = outputs[0];

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto task = [in = in, out = out, group = group()]() mutable {
    if (in.event().valid()) {
      in.event().wait();
    }
    distributed::detail::all_gather(group, in, out);
    out.event().signal();
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));
  signal_and_wait(in, out);
}

void Send::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& in = inputs[0];
  auto& out = outputs[0];
  move_or_copy(in, out);

  // Schedule an async send on the comm stream
  auto task = [in = in, out = out, group = group(), dst = dst_]() mutable {
    if (in.event().valid()) {
      in.event().wait();
    }
    distributed::detail::send(group, out, dst);
    out.event().signal();
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));

  // Encode a signal event for the input but not a wait since we don't need to
  // wait on the output.
  if (in.event().valid()) {
    encode_signal(in.event());
  }
}

void Recv::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 0);
  assert(outputs.size() == 1);

  auto& out = outputs[0];

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // Schedule an async recv on the comm stream
  auto task = [out = out, group = group(), src = src_]() mutable {
    distributed::detail::recv(group, out, src);
    out.event().signal();
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));

  // Encode a wait event as there is no input for the recv to encode a signal.
  encode_wait(out.event());
}

} // namespace mlx::core::distributed
