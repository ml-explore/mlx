// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename IdxT, int N_WRITES>
__global__ void arange(T* out, IdxT size, T start, T step) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_WRITES > size) {
    for (IdxT i = index * N_WRITES; i < size; ++i) {
      out[i] = start + i * step;
    }
  } else {
    AlignedVector<T, N_WRITES> out_vec;
#pragma unroll
    for (int i = 0; i < N_WRITES; ++i) {
      out_vec[i] = start + (index * N_WRITES + i) * step;
    }

    store_vector<N_WRITES>(out, index, out_vec);
  }
}

} // namespace cu

// ============================================================================
// Global volatile pointers for Windows/MSVC kernel registration
// ============================================================================
#ifdef _MSC_VER
// arange kernels for all int/float types with N_WRITES computed from sizeof(T)
// bool has N_WRITES=16, int8/uint8 has N_WRITES=16
volatile void* g_arange_bool_i32_16 =
    reinterpret_cast<void*>(&cu::arange<bool, int32_t, 16>);
volatile void* g_arange_i8_i32_16 =
    reinterpret_cast<void*>(&cu::arange<int8_t, int32_t, 16>);
volatile void* g_arange_u8_i32_16 =
    reinterpret_cast<void*>(&cu::arange<uint8_t, int32_t, 16>);
// int16/uint16/half/bf16 have N_WRITES=8
volatile void* g_arange_i16_i32_8 =
    reinterpret_cast<void*>(&cu::arange<int16_t, int32_t, 8>);
volatile void* g_arange_u16_i32_8 =
    reinterpret_cast<void*>(&cu::arange<uint16_t, int32_t, 8>);
volatile void* g_arange_half_i32_8 =
    reinterpret_cast<void*>(&cu::arange<__half, int32_t, 8>);
volatile void* g_arange_bf16_i32_8 =
    reinterpret_cast<void*>(&cu::arange<__nv_bfloat16, int32_t, 8>);
// int32/uint32/float32 have N_WRITES=4
volatile void* g_arange_i32_i32_4 =
    reinterpret_cast<void*>(&cu::arange<int32_t, int32_t, 4>);
volatile void* g_arange_u32_i32_4 =
    reinterpret_cast<void*>(&cu::arange<uint32_t, int32_t, 4>);
volatile void* g_arange_float_i32_4 =
    reinterpret_cast<void*>(&cu::arange<float, int32_t, 4>);
// int64/uint64/float64 have N_WRITES=2
volatile void* g_arange_i64_i32_2 =
    reinterpret_cast<void*>(&cu::arange<int64_t, int32_t, 2>);
volatile void* g_arange_u64_i32_2 =
    reinterpret_cast<void*>(&cu::arange<uint64_t, int32_t, 2>);
volatile void* g_arange_double_i32_2 =
    reinterpret_cast<void*>(&cu::arange<double, int32_t, 2>);

// int64 index versions (for very large arrays)
volatile void* g_arange_i32_i64_4 =
    reinterpret_cast<void*>(&cu::arange<int32_t, int64_t, 4>);
volatile void* g_arange_float_i64_4 =
    reinterpret_cast<void*>(&cu::arange<float, int64_t, 4>);
volatile void* g_arange_i64_i64_2 =
    reinterpret_cast<void*>(&cu::arange<int64_t, int64_t, 2>);
volatile void* g_arange_double_i64_2 =
    reinterpret_cast<void*>(&cu::arange<double, int64_t, 2>);

// Runtime kernel selector
void* get_arange_kernel(Dtype dtype, bool large) {
  if (!large) {
    switch (dtype.val()) {
      case bool_:
        return const_cast<void*>(g_arange_bool_i32_16);
      case int8:
        return const_cast<void*>(g_arange_i8_i32_16);
      case uint8:
        return const_cast<void*>(g_arange_u8_i32_16);
      case int16:
        return const_cast<void*>(g_arange_i16_i32_8);
      case uint16:
        return const_cast<void*>(g_arange_u16_i32_8);
      case float16:
        return const_cast<void*>(g_arange_half_i32_8);
      case bfloat16:
        return const_cast<void*>(g_arange_bf16_i32_8);
      case int32:
        return const_cast<void*>(g_arange_i32_i32_4);
      case uint32:
        return const_cast<void*>(g_arange_u32_i32_4);
      case float32:
        return const_cast<void*>(g_arange_float_i32_4);
      case int64:
        return const_cast<void*>(g_arange_i64_i32_2);
      case uint64:
        return const_cast<void*>(g_arange_u64_i32_2);
      case float64:
        return const_cast<void*>(g_arange_double_i32_2);
      default:
        return nullptr;
    }
  } else {
    switch (dtype.val()) {
      case int32:
        return const_cast<void*>(g_arange_i32_i64_4);
      case float32:
        return const_cast<void*>(g_arange_float_i64_4);
      case int64:
        return const_cast<void*>(g_arange_i64_i64_2);
      case float64:
        return const_cast<void*>(g_arange_double_i64_2);
      default:
        return nullptr;
    }
  }
}
#endif // _MSC_VER

// Helper template function to work around MSVC template function pointer
// issues. By making OutType, IdxT, and N_WRITES explicit template parameters of
// this function rather than deduced from nested lambdas, MSVC can resolve the
// kernel function pointer type correctly.
//
// IMPORTANT: We use explicit void* params[] instead of the variadic template
// because the variadic template stores pointers to temporaries that may go
// out of scope before the kernel runs (since kernels are queued in a graph).
template <typename OutType, typename IdxT, int N_WRITES>
void launch_arange_kernel(
    cu::CommandEncoder& encoder,
    array& out,
    double start,
    double step) {
  auto [num_blocks, block_dims] =
      get_launch_args(out, sizeof(IdxT) == sizeof(int64_t), N_WRITES);
  auto kernel = reinterpret_cast<void*>(&cu::arange<OutType, IdxT, N_WRITES>);

  // Store parameters in variables to ensure they remain valid until kernel
  // runs. The variadic template add_kernel_node stores pointers to its
  // arguments, so passing rvalues/temporaries causes undefined behavior.
  OutType* out_ptr = gpu_ptr<OutType>(out);
  IdxT size = static_cast<IdxT>(out.data_size());
  OutType start_val = static_cast<OutType>(start);
  OutType step_val =
      static_cast<OutType>(start + step) - static_cast<OutType>(start);

  void* params[] = {&out_ptr, &size, &start_val, &step_val};
  encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
}

// Dispatch helper for N_WRITES values based on element size.
// This avoids computing N_WRITES inside nested lambdas, which MSVC can't
// handle as a constexpr template argument.
template <typename OutType, typename IdxT>
void dispatch_arange(
    cu::CommandEncoder& encoder,
    array& out,
    double start,
    double step) {
  constexpr int N_WRITES = 16 / sizeof(OutType);
  launch_arange_kernel<OutType, IdxT, N_WRITES>(encoder, out, start, step);
}

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Arange::eval_gpu");
  if (out.size() == 0) {
    return;
  }
  auto& encoder = cu::get_command_encoder(stream());
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  encoder.set_output_array(out);

#ifdef _MSC_VER
  // On Windows, use runtime-selected registered kernel
  bool large = out.data_size() > INT32_MAX;
  void* kernel = get_arange_kernel(out.dtype(), large);
  if (kernel != nullptr) {
    constexpr int N_WRITES_1 = 16; // for sizeof=1
    constexpr int N_WRITES_2 = 8; // for sizeof=2
    constexpr int N_WRITES_4 = 4; // for sizeof=4
    constexpr int N_WRITES_8 = 2; // for sizeof=8

    int n_writes = N_WRITES_4; // default for most types
    size_t elem_size = size_of(out.dtype());
    if (elem_size == 1)
      n_writes = N_WRITES_1;
    else if (elem_size == 2)
      n_writes = N_WRITES_2;
    else if (elem_size == 8)
      n_writes = N_WRITES_8;

    auto [num_blocks, block_dims] = get_launch_args(out, large, n_writes);

    // We need to convert start/step to the output type before passing
    // This is tricky because we need type-specific conversions
    // Use a union to store the parameters
    union {
      int32_t i32;
      int64_t i64;
      float f32;
      double f64;
      uint32_t u32;
      uint64_t u64;
      int16_t i16;
      uint16_t u16;
      int8_t i8;
      uint8_t u8;
      __half f16;
      __nv_bfloat16 bf16;
      bool b;
    } start_val, step_val;

    // Set values based on dtype
    switch (out.dtype().val()) {
      case bool_:
        start_val.b = static_cast<bool>(start_);
        step_val.b = static_cast<bool>(step_);
        break;
      case int8:
        start_val.i8 = static_cast<int8_t>(start_);
        step_val.i8 =
            static_cast<int8_t>(start_ + step_) - static_cast<int8_t>(start_);
        break;
      case uint8:
        start_val.u8 = static_cast<uint8_t>(start_);
        step_val.u8 =
            static_cast<uint8_t>(start_ + step_) - static_cast<uint8_t>(start_);
        break;
      case int16:
        start_val.i16 = static_cast<int16_t>(start_);
        step_val.i16 =
            static_cast<int16_t>(start_ + step_) - static_cast<int16_t>(start_);
        break;
      case uint16:
        start_val.u16 = static_cast<uint16_t>(start_);
        step_val.u16 = static_cast<uint16_t>(start_ + step_) -
            static_cast<uint16_t>(start_);
        break;
      case float16:
        start_val.f16 = __float2half(static_cast<float>(start_));
        step_val.f16 = __float2half(static_cast<float>(start_ + step_)) -
            __float2half(static_cast<float>(start_));
        break;
      case bfloat16:
        start_val.bf16 = __float2bfloat16(static_cast<float>(start_));
        step_val.bf16 = __float2bfloat16(static_cast<float>(start_ + step_)) -
            __float2bfloat16(static_cast<float>(start_));
        break;
      case int32:
        start_val.i32 = static_cast<int32_t>(start_);
        step_val.i32 =
            static_cast<int32_t>(start_ + step_) - static_cast<int32_t>(start_);
        break;
      case uint32:
        start_val.u32 = static_cast<uint32_t>(start_);
        step_val.u32 = static_cast<uint32_t>(start_ + step_) -
            static_cast<uint32_t>(start_);
        break;
      case float32:
        start_val.f32 = static_cast<float>(start_);
        step_val.f32 =
            static_cast<float>(start_ + step_) - static_cast<float>(start_);
        break;
      case int64:
        start_val.i64 = static_cast<int64_t>(start_);
        step_val.i64 =
            static_cast<int64_t>(start_ + step_) - static_cast<int64_t>(start_);
        break;
      case uint64:
        start_val.u64 = static_cast<uint64_t>(start_);
        step_val.u64 = static_cast<uint64_t>(start_ + step_) -
            static_cast<uint64_t>(start_);
        break;
      case float64:
        start_val.f64 = start_;
        step_val.f64 = step_;
        break;
      default:
        throw std::runtime_error("Unsupported dtype for arange");
    }

    // Must use gpu_ptr, not data<void>() which may copy to managed/host memory
    void* out_ptr = gpu_ptr<void>(out);
    if (large) {
      int64_t size_param = static_cast<int64_t>(out.data_size());
      void* params[] = {&out_ptr, &size_param, &start_val, &step_val};
      encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
    } else {
      int32_t size_param = static_cast<int32_t>(out.data_size());
      void* params[] = {&out_ptr, &size_param, &start_val, &step_val};
      encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
    }
    return;
  }
#endif

  dispatch_int_float_types(out.dtype(), "Arange", [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    using OutType = cuda_type_t<CTYPE>;
    dispatch_bool(out.data_size() > INT32_MAX, [&](auto large) {
      using IdxT = std::conditional_t<large(), int64_t, int32_t>;
      dispatch_arange<OutType, IdxT>(encoder, out, start_, step_);
    });
  });
}

} // namespace mlx::core
