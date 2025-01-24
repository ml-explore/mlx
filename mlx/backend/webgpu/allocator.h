// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/device.h"

#include <betann/betann.h>

namespace mlx::core {
class array;
struct Dtype;
} // namespace mlx::core

namespace mlx::core::webgpu {

using allocator::Buffer;

// Holds data for both CPU and GPU.
class DoubleBuffer {
 public:
  // Allocates memory in CPU.
  explicit DoubleBuffer(size_t size);
  // Allocates memory in GPU.
  DoubleBuffer(betann::Device& device, Dtype dtype, size_t size);

  ~DoubleBuffer();

  void set_cpu_data(void* data) {
    assert(!cpu_data_);
    cpu_data_ = data;
  }
  void set_gpu_data(betann::Buffer buffer) {
    gpu_data_ = std::move(buffer);
  }

  void* cpu_data() const {
    return cpu_data_;
  }
  const betann::Buffer& gpu_data() const {
    return gpu_data_;
  }

  size_t size() const {
    return size_;
  }

 private:
  size_t size_;
  void* cpu_data_ = nullptr;
  betann::Buffer gpu_data_;
};

class WgpuAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size, bool allow_swap) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  void ensure_cpu_data(array& arr, const void* data);
  void ensure_gpu_data(array& arr);
  Buffer malloc_gpu(array& arr);
  Buffer malloc_gpu(array& arr, size_t size);

 private:
  WgpuAllocator();
  friend WgpuAllocator& allocator();

  betann::Device& device_;
};

WgpuAllocator& allocator();

betann::Device& device(mlx::core::Device);
betann::Device& device(array& arr);

} // namespace mlx::core::webgpu
