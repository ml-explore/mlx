// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace {

template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t N) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

void hip_host_free_callback(void* ptr) {
  (void)hipHostFree(ptr);
}

} // namespace

namespace mlx::core {

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = rocm::get_command_encoder(stream());
  auto size = out.size();
  auto nbytes = size * out.itemsize();
  out.set_data(allocator::malloc(nbytes));
  // Stage through PINNED host memory. An async H2D copy from pageable memory is
  // unreliable on a discrete GPU over a non-coherent link (TB5 eGPU): the driver
  // must internally stage it, which can stall the stream (queue stuck, GPU shows
  // busy, the eval's sync never returns). Pinned memory DMAs directly and lets
  // the copy actually run asynchronously.
  void* out_ptr = nullptr;
  if (hipHostMalloc(&out_ptr, nbytes, hipHostMallocDefault) != hipSuccess ||
      out_ptr == nullptr) {
    // Fallback: pageable + synchronous copy (still correct, just slower).
    out_ptr = malloc(nbytes);
    reader_->read(static_cast<char*>(out_ptr), nbytes, offset_);
    if (swap_endianness_) {
      switch (out.itemsize()) {
        case 2: swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), size); break;
        case 4: swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), size); break;
        case 8: swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), size); break;
      }
    }
    (void)hipMemcpy(gpu_ptr<void>(out), out_ptr, nbytes, hipMemcpyHostToDevice);
    free(out_ptr);
    return;
  }
  reader_->read(static_cast<char*>(out_ptr), nbytes, offset_);
  if (swap_endianness_) {
    switch (out.itemsize()) {
      case 2:
        swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), size);
        break;
      case 4:
        swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), size);
        break;
      case 8:
        swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), size);
        break;
    }
  }
  (void)hipMemcpyAsync(
      gpu_ptr<void>(out),
      out_ptr,
      nbytes,
      hipMemcpyHostToDevice,
      encoder.stream());
  (void)hipLaunchHostFunc(encoder.stream(), hip_host_free_callback, out_ptr);
}

} // namespace mlx::core
