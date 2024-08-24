// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/ops.h"
#include "mlx/scheduler.h"

// TODO consider moving gpu functionality to metal/
#include "mlx/backend/metal/device.h"

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

std::pair<std::vector<array>, std::vector<int>> AllReduce::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  switch (reduce_type_) {
    case Sum:
      return {{all_sum(inputs[0], group())}, axes};
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
      return {all_sum(tangents[0], group())};
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
  return {{all_gather(inputs[0], group())}, axes};
}

std::vector<array> AllGather::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {all_gather(tangents[0], group())};
}

std::vector<array> AllGather::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto g = group();
  std::vector<int> starts(primals[0].ndim(), 0);
  auto stops = primals[0].shape();
  starts[0] = g.rank() * stops[0];
  stops[0] += starts[0];
  return {slice(cotangents[0], starts, stops)};
}

} // namespace mlx::core::distributed
