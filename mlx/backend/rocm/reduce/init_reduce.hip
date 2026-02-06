// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename T, typename U, typename Op>
__global__ void init_reduce_kernel(U* out, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    out[index] = ReduceInit<Op, T>::value();
  }
}

} // namespace rocm

// Dispatch reduce operations
template <typename F>
void dispatch_reduce_ops_init(Reduce::ReduceType reduce_type, F&& f) {
  switch (reduce_type) {
    case Reduce::Sum:
      f(type_identity<rocm::Sum>{});
      break;
    case Reduce::Prod:
      f(type_identity<rocm::Prod>{});
      break;
    case Reduce::Max:
      f(type_identity<rocm::Max>{});
      break;
    case Reduce::Min:
      f(type_identity<rocm::Min>{});
      break;
    case Reduce::And:
      f(type_identity<rocm::And>{});
      break;
    case Reduce::Or:
      f(type_identity<rocm::Or>{});
      break;
    default:
      throw std::runtime_error("Unsupported reduce type");
  }
}

void init_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  // Allocate if needed
  if (out.data_shared_ptr() == nullptr) {
    out.set_data(allocator::malloc(out.nbytes()));
  }
  
  encoder.set_output_array(out);
  
  int block_size = 256;
  int num_blocks = (out.size() + block_size - 1) / block_size;
  
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops_init(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename rocm::ReduceResult<OP, T>::type;
      
      encoder.launch_kernel([&](hipStream_t stream) {
        hipLaunchKernelGGL(
            (rocm::init_reduce_kernel<T, U, OP>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            out.data<U>(), out.size());
      });
    });
  });
}

} // namespace mlx::core
