// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/utils.h"

#include <cstdint>

namespace mlx::core {

namespace cu {

__global__ void token_offset_kernel(
    const uint32_t* indices,
    int64_t size,
    int group_count,
    int32_t* offsets) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= group_count) {
    return;
  }
  int64_t lo = 0;
  int64_t hi = size;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (indices[mid] < static_cast<uint32_t>(g)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  offsets[g] = static_cast<int32_t>(lo);
}

} // namespace cu

array compute_token_offset(
    const array& indices,
    int group_count,
    cu::CommandEncoder& encoder) {
  array offsets(
      cu::malloc_async(group_count * sizeof(int32_t), encoder),
      {group_count, 1, 1}, // 3D broadcasting required by cudnn
      int32);
  encoder.add_temporary(offsets);

  encoder.set_input_array(indices);
  encoder.set_output_array(offsets);

  int block = 256;
  int grid = (group_count + block - 1) / block;
  encoder.add_kernel_node(
      cu::token_offset_kernel,
      grid,
      block,
      gpu_ptr<uint32_t>(indices),
      static_cast<int64_t>(indices.size()),
      group_count,
      gpu_ptr<int32_t>(offsets));
  return offsets;
}

} // namespace mlx::core
