// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_s(const In* in, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = cast_to<Out>(in[0]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = cast_to<Out>(in[0]);
      }
    }
  }
}

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_v(const In* in, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = cast_to<Out>(in[i + j]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = cast_to<Out>(in[j]);
      }
    }
  }
}

} // namespace rocm

void copy_contiguous(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset) {
  
  // Handle empty arrays
  size_t size = out.data_size();
  if (size == 0) {
    return;
  }
  
  // Debug: Check if buffers are valid
  auto* in_buf = static_cast<const rocm::RocmBuffer*>(in.buffer().ptr());
  auto* out_buf = static_cast<rocm::RocmBuffer*>(out.buffer().ptr());
  
  if (!in_buf || !in_buf->data) {
    fprintf(stderr, "copy_contiguous: input buffer is null! in_buf=%p, in_buf->data=%p\n",
            (void*)in_buf, in_buf ? in_buf->data : nullptr);
    return;
  }
  if (!out_buf || !out_buf->data) {
    fprintf(stderr, "copy_contiguous: output buffer is null! out_buf=%p, out_buf->data=%p\n",
            (void*)out_buf, out_buf ? out_buf->data : nullptr);
    return;
  }
  
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
        using InType = hip_type_t<MLX_GET_TYPE(in_type_tag)>;
        using OutType = hip_type_t<MLX_GET_TYPE(out_type_tag)>;
        using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
        constexpr int N_READS = 4;
        
        int block_size = 256;
        int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
        num_blocks = std::max(1, std::min(num_blocks, 65535));
        
        const InType* in_ptr = gpu_ptr<const InType>(in) + in_offset;
        OutType* out_ptr = gpu_ptr<OutType>(out) + out_offset;
        
        encoder.launch_kernel([&](hipStream_t stream) {
          if (ctype == CopyType::Scalar) {
            hipLaunchKernelGGL(
                (rocm::copy_s<InType, OutType, IdxT, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                in_ptr, out_ptr, static_cast<IdxT>(size));
          } else {
            hipLaunchKernelGGL(
                (rocm::copy_v<InType, OutType, IdxT, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                in_ptr, out_ptr, static_cast<IdxT>(size));
          }
        });
      });
    });
  });
}

} // namespace mlx::core
