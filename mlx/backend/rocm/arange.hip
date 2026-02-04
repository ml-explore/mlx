// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
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
  
  out.set_data(allocator::malloc(out.nbytes()));
  
  size_t size = out.size();
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (out.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            rocm::arange_kernel<float>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            out.data<float>(), static_cast<float>(start_), static_cast<float>(step_), size);
        break;
      case float64:
        hipLaunchKernelGGL(
            rocm::arange_kernel<double>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            out.data<double>(), start_, step_, size);
        break;
      case float16:
        hipLaunchKernelGGL(
            rocm::arange_kernel<__half>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            out.data<__half>(), __float2half(static_cast<float>(start_)), __float2half(static_cast<float>(step_)), size);
        break;
      case bfloat16:
        hipLaunchKernelGGL(
            rocm::arange_kernel<hip_bfloat16>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            out.data<hip_bfloat16>(), hip_bfloat16(static_cast<float>(start_)), hip_bfloat16(static_cast<float>(step_)), size);
        break;
      case int32:
        hipLaunchKernelGGL(
            rocm::arange_kernel<int32_t>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            out.data<int32_t>(), static_cast<int32_t>(start_), static_cast<int32_t>(step_), size);
        break;
      case int64:
        hipLaunchKernelGGL(
            rocm::arange_kernel<int64_t>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            out.data<int64_t>(), static_cast<int64_t>(start_), static_cast<int64_t>(step_), size);
        break;
      default:
        throw std::runtime_error("Unsupported type for arange");
    }
  });
}

} // namespace mlx::core
