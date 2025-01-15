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

void signal_and_wait(const Event& e_signal, const Event& e_wait) {
  if (e_signal.valid()) {
    encode_signal(e_signal);
  }
  encode_wait(e_wait);
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

  auto e = Event(stream());
  e.set_value(1);
  signal_and_wait(in.event(), e);
  auto task = [in = in,
               out = out,
               e = std::move(e),
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
    e.signal();
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& out = outputs[0];

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto e = Event(stream());
  e.set_value(1);
  signal_and_wait(in.event(), e);

  auto task =
      [in = in, out = out, e = std::move(e), group = group()]() mutable {
        if (in.event().valid()) {
          in.event().wait();
        }
        distributed::detail::all_gather(group, in, out);
        e.signal();
      };
  scheduler::enqueue(detail::communication_stream(), std::move(task));
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
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));

  // Encode a signal event for the input
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

  auto e = Event(stream());
  e.set_value(1);

  encode_wait(e);

  // Schedule an async recv on the comm stream
  auto task =
      [out = out, e = std::move(e), group = group(), src = src_]() mutable {
        distributed::detail::recv(group, out, src);
        e.signal();
      };
  scheduler::enqueue(detail::communication_stream(), std::move(task));
}

} // namespace mlx::core::distributed
