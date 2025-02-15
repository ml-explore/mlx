// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"

namespace mlx::core::distributed {

namespace {
array ensure_row_contiguous(const array& arr, Stream s) {
  if (arr.flags().row_contiguous) {
    return arr;
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy_gpu(arr, arr_copy, CopyType::General, s);
    return arr_copy;
  }
}
} // namespace

void AllReduce::eval_gpu(
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
      copy_gpu(in, arr_copy, CopyType::General, s);
      out.copy_shared_buffer(arr_copy);
      return arr_copy;
    }
  };

  auto in = donate_or_copy(inputs[0], outputs[0]);
  auto& out = outputs[0];
  /*
      Fence f{stream()};

      if (in.event().valid()) {
        f.increment();
        f.update_gpu(in);
      }

      if (in.is_donatable()) {
        out.move_shared_buffer(in);
      } else {
        out.set_data(allocator::malloc_or_wait(out.nbytes()));
      }
      f.wait_gpu(out);

      auto task = [in = in,
                   out = unsafe_weak_copy(out),
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
            throw std::runtime_error("Only all reduce sum is supported for
     now");
        }
        f.update();
      };
      scheduler::enqueue(detail::communication_stream(), std::move(task));*/
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
  auto in = ensure_row_contiguous(inputs[0], stream());
  auto& out = outputs[0];
  /*out.set_data(allocator::malloc_or_wait(out.nbytes()));

  Fence f{stream()};

  if (in.event().valid()) {
    f.update_gpu(in);
  }
  f.wait_gpu(out);

  auto task = [in = in,
               out = unsafe_weak_copy(out),
               f = std::move(f),
               group = group()]() mutable {
    if (in.event().valid()) {
      f.wait();
    }
    distributed::detail::all_gather(group, in, out);
    f.update();
  };
  scheduler::enqueue(detail::communication_stream(), std::move(task));*/
}

void Send::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto in = ensure_row_contiguous(inputs[0], stream());
  /*
      // Encode a signal event for the input
      Fence f{stream()};
      if (in.event().valid()) {
        f.update_gpu(in);
      }

      auto& out = outputs[0];
      out.copy_shared_buffer(in, out);

      // Schedule an async send on the comm stream
      auto task = [in = in,
                   out = unsafe_weak_copy(out),
                   f = std::move(f),
                   group = group(),
                   dst = dst_]() mutable {
        if (in.event().valid()) {
          f.wait();
        }
        distributed::detail::send(group, out, dst);
      };
      scheduler::enqueue(detail::communication_stream(), std::move(task));*/
}

void Recv::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  /*  assert(inputs.size() == 0);
    assert(outputs.size() == 1);

    auto& out = outputs[0];

    out.set_data(allocator::malloc_or_wait(out.nbytes()));

    Fence f{stream()};
    f.wait_gpu(out);

    // Schedule an async recv on the comm stream
    auto task = [out = unsafe_weak_copy(out),
                 f = std::move(f),
                 group = group(),
                 src = src_]() mutable {
      distributed::detail::recv(group, out, src);
      f.update();
    };
    scheduler::enqueue(detail::communication_stream(), std::move(task));*/
}

} // namespace mlx::core::distributed
