// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/event.h"
#include "mlx/backend/metal/fence.h"
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

  Fence f{stream()};

  if (in.event().valid()) {
    f.update_gpu(in);
  }

  auto& out = outputs[0];
  if (in.is_donatable()) {
    out.move_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }
  f.wait_gpu(out);

  auto task = [in = in,
               out = out,
               f = std::move(f),
               reduce_type = reduce_type_,
               group = group()]() mutable {
    if (in.event().valid()) {
      f.wait();
    }
    switch (reduce_type) {
      case Sum:
        distributed::detail::all_sum(
            group, in.data_shared_ptr() == nullptr ? out : in, out);
        break;
      default:
        throw std::runtime_error("Only all reduce sum is supported for now");
    }
    f.update();
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

  Fence f{stream()};

  if (in.event().valid()) {
    f.update_gpu(in);
  }
  f.wait_gpu(out);

  auto task =
      [in = in, out = out, f = std::move(f), group = group()]() mutable {
        if (in.event().valid()) {
          f.wait();
        }
        distributed::detail::all_gather(group, in, out);
        f.update();
      };
  scheduler::enqueue(detail::communication_stream(), std::move(task));
}

void Send::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& in = inputs[0];

  // Encode a signal event for the input
  Fence f{stream()};
  if (in.event().valid()) {
    f.update_gpu(in);
  }

  auto& out = outputs[0];
  move_or_copy(in, out);

  // Schedule an async send on the comm stream
  auto task = [in = in,
               out = out,
               f = std::move(f),
               group = group(),
               dst = dst_]() mutable {
    if (in.event().valid()) {
      f.wait();
    }
    distributed::detail::send(group, out, dst);
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));
}

void Recv::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 0);
  assert(outputs.size() == 1);

  auto& out = outputs[0];

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  Fence f{stream()};
  f.wait_gpu(out);

  // Schedule an async recv on the comm stream
  auto task =
      [out = out, f = std::move(f), group = group(), src = src_]() mutable {
        distributed::detail::recv(group, out, src);
        f.update();
      };
  scheduler::enqueue(detail::communication_stream(), std::move(task));
}

} // namespace mlx::core::distributed
