// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename U, typename Op>
__global__ void init_reduce(U* out, size_t size) {
  auto index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = ReduceInit<Op, T>::value();
  }
}

} // namespace cu

void init_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  // Allocate if needed
  if (out.data_shared_ptr() == nullptr) {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
  }

  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;
      auto kernel = cu::init_reduce<T, U, OP>;
      dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
      dim3 block(grid.x < 1024 ? grid.x : 1024, 1, 1);
      grid.x = (grid.x + 1023) / 1024;

      // Store params in variables to ensure they remain valid
      U* out_ptr = gpu_ptr<U>(out);
      size_t size = out.size();
      void* params[] = {&out_ptr, &size};
      encoder.add_kernel_node(
          reinterpret_cast<void*>(kernel), grid, block, 0, params);
    });
  });
}

} // namespace mlx::core

// Force instantiation of kernel templates for Windows MSVC/NVCC compatibility.
// Global volatile pointers force the addresses to be computed at compile time,
// which triggers NVCC's kernel registration code generation.
#ifdef _MSC_VER
using namespace mlx::core::cu;

volatile void* g_init_reduce_and =
    reinterpret_cast<void*>(&init_reduce<bool, bool, And>);
volatile void* g_init_reduce_or =
    reinterpret_cast<void*>(&init_reduce<bool, bool, Or>);
volatile void* g_init_reduce_sum_bool =
    reinterpret_cast<void*>(&init_reduce<bool, int32_t, Sum>);
volatile void* g_init_reduce_sum_i32 =
    reinterpret_cast<void*>(&init_reduce<int32_t, int32_t, Sum>);
volatile void* g_init_reduce_sum_i64 =
    reinterpret_cast<void*>(&init_reduce<int64_t, int64_t, Sum>);
volatile void* g_init_reduce_sum_u32 =
    reinterpret_cast<void*>(&init_reduce<uint32_t, uint32_t, Sum>);
volatile void* g_init_reduce_sum_u64 =
    reinterpret_cast<void*>(&init_reduce<uint64_t, uint64_t, Sum>);
volatile void* g_init_reduce_sum_float =
    reinterpret_cast<void*>(&init_reduce<float, float, Sum>);
volatile void* g_init_reduce_sum_half =
    reinterpret_cast<void*>(&init_reduce<__half, float, Sum>);
volatile void* g_init_reduce_sum_bf16 =
    reinterpret_cast<void*>(&init_reduce<__nv_bfloat16, float, Sum>);
volatile void* g_init_reduce_prod_i32 =
    reinterpret_cast<void*>(&init_reduce<int32_t, int32_t, Prod>);
volatile void* g_init_reduce_prod_float =
    reinterpret_cast<void*>(&init_reduce<float, float, Prod>);
volatile void* g_init_reduce_max_i32 =
    reinterpret_cast<void*>(&init_reduce<int32_t, int32_t, Max>);
volatile void* g_init_reduce_max_float =
    reinterpret_cast<void*>(&init_reduce<float, float, Max>);
volatile void* g_init_reduce_max_half =
    reinterpret_cast<void*>(&init_reduce<__half, __half, Max>);
volatile void* g_init_reduce_max_bf16 =
    reinterpret_cast<void*>(&init_reduce<__nv_bfloat16, __nv_bfloat16, Max>);
volatile void* g_init_reduce_min_i32 =
    reinterpret_cast<void*>(&init_reduce<int32_t, int32_t, Min>);
volatile void* g_init_reduce_min_float =
    reinterpret_cast<void*>(&init_reduce<float, float, Min>);
volatile void* g_init_reduce_min_half =
    reinterpret_cast<void*>(&init_reduce<__half, __half, Min>);
volatile void* g_init_reduce_min_bf16 =
    reinterpret_cast<void*>(&init_reduce<__nv_bfloat16, __nv_bfloat16, Min>);
#endif // _MSC_VER
