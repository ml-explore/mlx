// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"

#include <mutex>
#include <set>
#include <thread>
#include <utility>

namespace mlx::core::cu {

class Worker;

using allocator::Buffer;

// Stores cuda-managed unified memory.
struct CudaBuffer {
  void* data;
  size_t size;
};

class CudaAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  // Register current thread as safe to free buffers.
  // In cuda freeing a buffer implicitly synchronizes stream, and for threads
  // that may be waited by gpu stream (for example cpu stream threads), freeing
  // buffers there would result in dead lock.
  void register_this_thread();

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);

 private:
  CudaAllocator();
  friend CudaAllocator& allocator();

  std::mutex worker_mutex_;
  std::unique_ptr<Worker> worker_;
  std::set<std::thread::id> allowed_threads_;

  std::mutex mutex_;
  size_t memory_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
};

CudaAllocator& allocator();

} // namespace mlx::core::cu
