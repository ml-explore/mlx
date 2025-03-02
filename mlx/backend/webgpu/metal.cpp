// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/metal_impl.h"
#include "mlx/backend/webgpu/allocator.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
}

void new_stream(Stream s) {
  static bool has_gpu_stream = false;
  if (s.device == Device::gpu) {
    if (has_gpu_stream) {
      throw std::runtime_error("WebGPU backend can not have multiple streams.");
    }
    has_gpu_stream = true;
  }
}

std::function<void()> make_task(array arr, bool signal) {
  return [arr = std::move(arr), signal]() mutable {
    auto s = arr.primitive().stream();
    auto& device = webgpu::device(s.device);

    for (auto& input : arr.inputs()) {
      if (input.event().valid() &&
          input.event().stream() != arr.primitive().stream()) {
        input.event().wait();
      }
      // Ensure all inputs copy their CPU data to GPU.
      webgpu::allocator().ensure_gpu_data(input);
    }

    auto outputs = arr.outputs();
    {
      std::vector<array> inputs;
      if (arr.is_tracer()) {
        inputs = arr.inputs();
      }

      try {
        arr.primitive().eval_gpu(arr.inputs(), outputs);
      } catch (const std::exception& error) {
        abort_with_exception(error);
      }
    }
    std::vector<std::shared_ptr<array::Data>> buffers;
    for (auto& in : arr.inputs()) {
      buffers.push_back(in.data_shared_ptr());
    }
    for (auto& s : arr.siblings()) {
      buffers.push_back(s.data_shared_ptr());
    }
    if (!arr.is_tracer()) {
      arr.detach();
    }
    for (auto& out : outputs) {
      out.set_status(array::Status::evaluated);
    }

    // Copy data from GPU to CPU.
    // FIXME(zcbenz): Should only do it when necessary.
    if (arr.data_shared_ptr()) {
      auto* dbuf = static_cast<webgpu::DoubleBuffer*>(arr.buffer().ptr());
      if (!dbuf->cpu_data() && dbuf->gpu_data()) {
        device.Flush();
        device.ReadBuffer(
            dbuf->gpu_data(),
            [arr, dbuf, buffers = std::move(buffers)](
                const void* data, uint64_t size, uint64_t offset) mutable {
              webgpu::allocator().ensure_cpu_data(arr, data);
              arr.reset_data_ptr();
            });
      }
    }

    if (signal) {
      device.Flush();
      device.WaitAll();
      arr.event().signal();
    } else {
      device.OnSubmittedWorkDone([buffers = std::move(buffers)]() {});
    }
  };
}

std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p) {
  return [s, p = std::move(p)]() {
    auto& device = webgpu::device(s.device);
    device.WaitAll();
    p->set_value();
  };
}

void start_capture(std::string) {
  throw std::runtime_error("WebGPU backend does not support capture.");
}

void stop_capture() {
  throw std::runtime_error("WebGPU backend does not support capture.");
}

} // namespace mlx::core::metal
