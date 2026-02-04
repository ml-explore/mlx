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

// General copy kernel - strided input to contiguous output
template <typename In, typename Out, typename IdxT>
__global__ void copy_g(
    const In* in,
    Out* out,
    IdxT size,
    const int* shape,
    const int64_t* strides,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  
  // Compute input offset from linear index
  IdxT in_offset = 0;
  IdxT tmp = index;
  for (int i = ndim - 1; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    in_offset += coord * strides[i];
    tmp /= shape[i];
  }
  
  out[index] = cast_to<Out>(in[in_offset]);
}

// General copy kernel - strided input to strided output
template <typename In, typename Out, typename IdxT>
__global__ void copy_gg(
    const In* in,
    Out* out,
    IdxT size,
    const int* shape,
    const int64_t* strides_in,
    const int64_t* strides_out,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  
  // Compute input and output offsets from linear index
  IdxT in_offset = 0;
  IdxT out_offset = 0;
  IdxT tmp = index;
  for (int i = ndim - 1; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    in_offset += coord * strides_in[i];
    out_offset += coord * strides_out[i];
    tmp /= shape[i];
  }
  
  out[out_offset] = cast_to<Out>(in[in_offset]);
}

} // namespace rocm

void copy_contiguous(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset) {
  
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
        using InType = hip_type_t<MLX_GET_TYPE(in_type_tag)>;
        using OutType = hip_type_t<MLX_GET_TYPE(out_type_tag)>;
        using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
        constexpr int N_READS = 4;
        
        int block_size = 256;
        size_t size = out.data_size();
        int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
        num_blocks = std::min(num_blocks, 65535);
        
        const InType* in_ptr = reinterpret_cast<const InType*>(in.data<void>()) + in_offset;
        OutType* out_ptr = reinterpret_cast<OutType*>(out.data<void>()) + out_offset;
        
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

void copy_general_input(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset,
    const Shape& shape,
    const Strides& strides_in) {
  
  int ndim = shape.size();
  
  // Allocate device memory for shape and strides
  std::vector<int> shape_int(shape.begin(), shape.end());
  
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
        using InType = hip_type_t<MLX_GET_TYPE(in_type_tag)>;
        using OutType = hip_type_t<MLX_GET_TYPE(out_type_tag)>;
        using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
        
        int block_size = 256;
        size_t size = out.data_size();
        int num_blocks = (size + block_size - 1) / block_size;
        num_blocks = std::min(num_blocks, 65535);
        
        const InType* in_ptr = reinterpret_cast<const InType*>(in.data<void>()) + in_offset;
        OutType* out_ptr = reinterpret_cast<OutType*>(out.data<void>()) + out_offset;
        
        encoder.launch_kernel([&](hipStream_t stream) {
          hipLaunchKernelGGL(
              (rocm::copy_g<InType, OutType, IdxT>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in_ptr, out_ptr, static_cast<IdxT>(size),
              shape_int.data(), strides_in.data(), ndim);
        });
      });
    });
  });
}

void copy_general(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out) {
  
  int ndim = shape.size();
  
  // Convert shape to int
  std::vector<int> shape_int(shape.begin(), shape.end());
  
  // Compute total size
  size_t size = 1;
  for (auto s : shape) size *= s;
  
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(in.data_size() > INT32_MAX || out.data_size() > INT32_MAX, [&](auto large) {
        using InType = hip_type_t<MLX_GET_TYPE(in_type_tag)>;
        using OutType = hip_type_t<MLX_GET_TYPE(out_type_tag)>;
        using IdxT = std::conditional_t<large(), int64_t, int32_t>;
        
        int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        num_blocks = std::min((size_t)num_blocks, (size_t)65535);
        
        const InType* in_ptr = reinterpret_cast<const InType*>(in.data<void>()) + in_offset;
        OutType* out_ptr = reinterpret_cast<OutType*>(out.data<void>()) + out_offset;
        
        encoder.launch_kernel([&](hipStream_t stream) {
          hipLaunchKernelGGL(
              (rocm::copy_gg<InType, OutType, IdxT>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in_ptr, out_ptr, static_cast<IdxT>(size),
              shape_int.data(), strides_in.data(), strides_out.data(), ndim);
        });
      });
    });
  });
}

} // namespace mlx::core
