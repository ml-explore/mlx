// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::distributed {

void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto& in = inputs[0];
  auto& out = outputs[0];
  if (in.is_donatable()) {
    // TODO should be a move?
    out.copy_shared_buffer(in);
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
        // TODO fix if move
        distributed::detail::all_sum(group, in, out);
        break;
      default:
        throw std::runtime_error("Only all reduce sum is supported for now");
    }
    out.event().signal();
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));

  // TODO clean this up and maybe make APIs for it
  auto& d = metal::device(stream().device);
  d.end_encoding(stream().index);
  auto command_buffer = d.get_command_buffer(stream().index);
  if (in.event().valid()) {
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(in.event().raw_event().get()),
        in.event().value());
  }
  command_buffer->encodeWait(
      static_cast<MTL::Event*>(out.event().raw_event().get()),
      out.event().value());
  // TODO Need to hold the event in a completion handler?
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

  auto& d = metal::device(stream().device);
  d.end_encoding(stream().index);
  auto command_buffer = d.get_command_buffer(stream().index);
  if (in.event().valid()) {
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(in.event().raw_event().get()),
        in.event().value());
  }
  command_buffer->encodeWait(
      static_cast<MTL::Event*>(out.event().raw_event().get()),
      out.event().value());
}

} // namespace mlx::core::distributed
