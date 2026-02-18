// Copyright © 2024 Apple Inc.

#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < size) {
    input[index] = input[index];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to update timestamp
[[kernel]] void fence_update(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to spin wait for timestamp value
[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  while (1) {
    // Fast path: volatile reads through GPU cache + system-scope fence
    for (uint i = 0; i < 1000000; i++) {
      metal::atomic_thread_fence(
          metal::mem_flags::mem_device,
          metal::memory_order_seq_cst,
          metal::thread_scope_system);
      if (timestamp[0] >= value) {
        return;
      }
    }
    // System-scope atomic load to force GPU cache refresh from SLC
    uint cur = __metal_atomic_load_explicit(
        timestamp,
        int(metal::memory_order_relaxed),
        __METAL_MEMORY_SCOPE_SYSTEM__);
    if (cur >= value) {
      return;
    }
  }
}
