// Copyright © 2023 Apple Inc.

#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

using allocator::Buffer;

class MetalAllocator : public allocator::Allocator {
  /** Allocator for Metal GPUs. */
 public:
  virtual Buffer malloc(size_t size, bool allow_swap = false) override;
  virtual void free(Buffer buffer) override;

 private:
  MTL::Device* device_;
  MetalAllocator();
  friend MetalAllocator& allocator();

  // Allocation stats
  size_t peak_allocated_size_;
  size_t block_limit_;
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
