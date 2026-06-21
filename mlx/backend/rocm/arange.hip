// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device/arange.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/primitives.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mlx::core {

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));

  size_t size = out.size();
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  switch (out.dtype()) {
    case float32: {
      float start = static_cast<float>(start_);
      float step = static_cast<float>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<float>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<float>(out), start, step, size);
      break;
    }
    case float64: {
      double start = start_;
      double step = step_;
      encoder.add_kernel_node(
          &rocm::arange_kernel<double>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<double>(out), start, step, size);
      break;
    }
    case float16: {
      __half start = __float2half(static_cast<float>(start_));
      __half step = __float2half(static_cast<float>(step_));
      encoder.add_kernel_node(
          &rocm::arange_kernel<__half>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<__half>(out), start, step, size);
      break;
    }
    case bfloat16: {
      hip_bfloat16 start = hip_bfloat16(static_cast<float>(start_));
      hip_bfloat16 step = hip_bfloat16(static_cast<float>(step_));
      encoder.add_kernel_node(
          &rocm::arange_kernel<hip_bfloat16>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<hip_bfloat16>(out), start, step, size);
      break;
    }
    case int32: {
      int32_t start = static_cast<int32_t>(start_);
      int32_t step = static_cast<int32_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<int32_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<int32_t>(out), start, step, size);
      break;
    }
    case int64: {
      int64_t start = static_cast<int64_t>(start_);
      int64_t step = static_cast<int64_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<int64_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<int64_t>(out), start, step, size);
      break;
    }
    case uint32: {
      uint32_t start = static_cast<uint32_t>(start_);
      uint32_t step = static_cast<uint32_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<uint32_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<uint32_t>(out), start, step, size);
      break;
    }
    case uint64: {
      uint64_t start = static_cast<uint64_t>(start_);
      uint64_t step = static_cast<uint64_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<uint64_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<uint64_t>(out), start, step, size);
      break;
    }
    case int8: {
      int8_t start = static_cast<int8_t>(start_);
      int8_t step = static_cast<int8_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<int8_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<int8_t>(out), start, step, size);
      break;
    }
    case int16: {
      int16_t start = static_cast<int16_t>(start_);
      int16_t step = static_cast<int16_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<int16_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<int16_t>(out), start, step, size);
      break;
    }
    case uint8: {
      uint8_t start = static_cast<uint8_t>(start_);
      uint8_t step = static_cast<uint8_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<uint8_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<uint8_t>(out), start, step, size);
      break;
    }
    case uint16: {
      uint16_t start = static_cast<uint16_t>(start_);
      uint16_t step = static_cast<uint16_t>(step_);
      encoder.add_kernel_node(
          &rocm::arange_kernel<uint16_t>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<uint16_t>(out), start, step, size);
      break;
    }
    default:
      throw std::runtime_error("Unsupported type for arange");
  }
}

} // namespace mlx::core
