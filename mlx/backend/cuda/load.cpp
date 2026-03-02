// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/primitives.h"

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

} // namespace

namespace mlx::core {

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = cu::get_command_encoder(stream());
  auto size = out.size();
  auto nbytes = size * out.itemsize();
  out.set_data(cu::malloc_async(nbytes, encoder));
  auto out_ptr = malloc(nbytes);
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
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      gpu_ptr<void>(out),
      out_ptr,
      nbytes,
      cudaMemcpyDefault,
      encoder.stream()));
  CHECK_CUDA_ERROR(cudaLaunchHostFunc(encoder.stream(), free, out_ptr));
}

} // namespace mlx::core
