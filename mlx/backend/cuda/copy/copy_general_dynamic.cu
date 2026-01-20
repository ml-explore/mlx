// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"
#include "mlx/backend/cuda/utils.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int NDIM>
__global__ void copy_gg_dynamic_nd(
    const In* in,
    Out* out,
    IdxT size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_in,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_out,
    const int64_t* offset_in,
    const int64_t* offset_out) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [idx_in, idx_out] = elem_to_loc_nd<NDIM>(
        index, shape.data(), strides_in.data(), strides_out.data());
    out[idx_out + *offset_out] = CastOp<In, Out>{}(in[idx_in + *offset_in]);
  }
}

template <typename In, typename Out, typename IdxT>
__global__ void copy_gg_dynamic(
    const In* in,
    Out* out,
    IdxT size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides strides_in,
    const __grid_constant__ Strides strides_out,
    int ndim,
    const int64_t* offset_in,
    const int64_t* offset_out) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [idx_in, idx_out] = elem_to_loc(
        index, shape.data(), strides_in.data(), strides_out.data(), ndim);
    out[idx_out + *offset_out] = CastOp<In, Out>{}(in[idx_in + *offset_in]);
  }
}

} // namespace cu

// ============================================================================
// Global volatile kernel pointers for Windows/MSVC kernel registration
// ============================================================================
// On Windows, MSVC doesn't properly resolve template types through nested
// lambdas with dispatch_all_types. These global volatile pointers force kernel
// instantiation and registration at static initialization time.

// copy_gg_dynamic_nd kernels - for DynamicSliceUpdate
// float: NDIM=1,2,3
volatile void* copy_gg_dyn_nd_float_1 =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic_nd<float, float, int32_t, 1>);
volatile void* copy_gg_dyn_nd_float_2 =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic_nd<float, float, int32_t, 2>);
volatile void* copy_gg_dyn_nd_float_3 =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic_nd<float, float, int32_t, 3>);

// int32: NDIM=1,2,3
volatile void* copy_gg_dyn_nd_i32_1 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<int32_t, int32_t, int32_t, 1>);
volatile void* copy_gg_dyn_nd_i32_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<int32_t, int32_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_i32_3 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<int32_t, int32_t, int32_t, 3>);

// bool: NDIM=1,2,3
volatile void* copy_gg_dyn_nd_bool_1 =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic_nd<bool, bool, int32_t, 1>);
volatile void* copy_gg_dyn_nd_bool_2 =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic_nd<bool, bool, int32_t, 2>);
volatile void* copy_gg_dyn_nd_bool_3 =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic_nd<bool, bool, int32_t, 3>);

// Additional types for NDIM=2 (most common)
volatile void* copy_gg_dyn_nd_i64_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<int64_t, int64_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_u32_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<uint32_t, uint32_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_half_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<__half, __half, int32_t, 2>);
volatile void* copy_gg_dyn_nd_bf16_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<__nv_bfloat16, __nv_bfloat16, int32_t, 2>);
volatile void* copy_gg_dyn_nd_double_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<double, double, int32_t, 2>);
volatile void* copy_gg_dyn_nd_c64_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<cu::complex64_t, cu::complex64_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_i8_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<int8_t, int8_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_u8_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<uint8_t, uint8_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_i16_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<int16_t, int16_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_u16_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<uint16_t, uint16_t, int32_t, 2>);
volatile void* copy_gg_dyn_nd_u64_2 = reinterpret_cast<void*>(
    &cu::copy_gg_dynamic_nd<uint64_t, uint64_t, int32_t, 2>);

// copy_gg_dynamic kernels (for ndim >= 4)
volatile void* copy_gg_dyn_float =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic<float, float, int32_t>);
volatile void* copy_gg_dyn_i32 =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic<int32_t, int32_t, int32_t>);
volatile void* copy_gg_dyn_bool =
    reinterpret_cast<void*>(&cu::copy_gg_dynamic<bool, bool, int32_t>);

// ============================================================================
// Runtime kernel selector for Windows/MSVC
// ============================================================================
#ifdef _MSC_VER
#include "mlx/dtype.h"

// Get the registered kernel for copy_gg_dynamic_nd based on runtime dtype and
// ndim
void* get_copy_gg_dynamic_nd_kernel_by_dtype(
    Dtype dtype,
    int ndim,
    bool large) {
  if (large)
    return nullptr;

  if (ndim == 1) {
    switch (dtype.val()) {
      case float32:
        return const_cast<void*>(copy_gg_dyn_nd_float_1);
      case int32:
        return const_cast<void*>(copy_gg_dyn_nd_i32_1);
      case bool_:
        return const_cast<void*>(copy_gg_dyn_nd_bool_1);
      default:
        break;
    }
  } else if (ndim == 2) {
    switch (dtype.val()) {
      case float32:
        return const_cast<void*>(copy_gg_dyn_nd_float_2);
      case int32:
        return const_cast<void*>(copy_gg_dyn_nd_i32_2);
      case bool_:
        return const_cast<void*>(copy_gg_dyn_nd_bool_2);
      case int64:
        return const_cast<void*>(copy_gg_dyn_nd_i64_2);
      case uint32:
        return const_cast<void*>(copy_gg_dyn_nd_u32_2);
      case float16:
        return const_cast<void*>(copy_gg_dyn_nd_half_2);
      case bfloat16:
        return const_cast<void*>(copy_gg_dyn_nd_bf16_2);
      case float64:
        return const_cast<void*>(copy_gg_dyn_nd_double_2);
      case complex64:
        return const_cast<void*>(copy_gg_dyn_nd_c64_2);
      case int8:
        return const_cast<void*>(copy_gg_dyn_nd_i8_2);
      case uint8:
        return const_cast<void*>(copy_gg_dyn_nd_u8_2);
      case int16:
        return const_cast<void*>(copy_gg_dyn_nd_i16_2);
      case uint16:
        return const_cast<void*>(copy_gg_dyn_nd_u16_2);
      case uint64:
        return const_cast<void*>(copy_gg_dyn_nd_u64_2);
      default:
        break;
    }
  } else if (ndim == 3) {
    switch (dtype.val()) {
      case float32:
        return const_cast<void*>(copy_gg_dyn_nd_float_3);
      case int32:
        return const_cast<void*>(copy_gg_dyn_nd_i32_3);
      case bool_:
        return const_cast<void*>(copy_gg_dyn_nd_bool_3);
      default:
        break;
    }
  }
  return nullptr;
}

// Get the registered kernel for copy_gg_dynamic based on runtime dtype
void* get_copy_gg_dynamic_kernel_by_dtype(Dtype dtype, bool large) {
  if (large)
    return nullptr;

  switch (dtype.val()) {
    case float32:
      return const_cast<void*>(copy_gg_dyn_float);
    case int32:
      return const_cast<void*>(copy_gg_dyn_i32);
    case bool_:
      return const_cast<void*>(copy_gg_dyn_bool);
    default:
      break;
  }
  return nullptr;
}
#endif // _MSC_VER

// Helper template functions to work around MSVC template function pointer
// issues. Cast to void* for MSVC compatibility - avoids template deduction
// issues.
template <typename InType, typename OutType, typename IdxT, int NDIM>
void launch_copy_gg_dynamic_nd_kernel(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t size,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const int64_t* offset_in,
    const int64_t* offset_out,
    dim3 num_blocks,
    dim3 block_dims) {
  auto kernel = reinterpret_cast<void*>(
      &cu::copy_gg_dynamic_nd<InType, OutType, IdxT, NDIM>);
  // Manually pack parameters to bypass MSVC variadic template deduction issues
  IdxT size_param = static_cast<IdxT>(size);
  auto shape_param = const_param<NDIM>(shape);
  auto strides_in_param = const_param<NDIM>(strides_in);
  auto strides_out_param = const_param<NDIM>(strides_out);
  void* params[] = {
      (void*)&in_ptr,
      (void*)&out_ptr,
      (void*)&size_param,
      (void*)&shape_param,
      (void*)&strides_in_param,
      (void*)&strides_out_param,
      (void*)&offset_in,
      (void*)&offset_out};
  encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
}

template <typename InType, typename OutType, typename IdxT>
void dispatch_copy_gg_dynamic_nd_ndim(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t size,
    int ndim,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const int64_t* offset_in,
    const int64_t* offset_out,
    dim3 num_blocks,
    dim3 block_dims) {
  switch (ndim) {
    case 1:
      launch_copy_gg_dynamic_nd_kernel<InType, OutType, IdxT, 1>(
          encoder,
          in_ptr,
          out_ptr,
          size,
          shape,
          strides_in,
          strides_out,
          offset_in,
          offset_out,
          num_blocks,
          block_dims);
      break;
    case 2:
      launch_copy_gg_dynamic_nd_kernel<InType, OutType, IdxT, 2>(
          encoder,
          in_ptr,
          out_ptr,
          size,
          shape,
          strides_in,
          strides_out,
          offset_in,
          offset_out,
          num_blocks,
          block_dims);
      break;
    case 3:
      launch_copy_gg_dynamic_nd_kernel<InType, OutType, IdxT, 3>(
          encoder,
          in_ptr,
          out_ptr,
          size,
          shape,
          strides_in,
          strides_out,
          offset_in,
          offset_out,
          num_blocks,
          block_dims);
      break;
  }
}

template <typename InType, typename OutType, typename IdxT>
void launch_copy_gg_dynamic_kernel(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t size,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int ndim,
    const int64_t* offset_in,
    const int64_t* offset_out,
    dim3 num_blocks,
    dim3 block_dims) {
  // Cast to void* for MSVC compatibility - avoids template deduction issues
  auto kernel =
      reinterpret_cast<void*>(&cu::copy_gg_dynamic<InType, OutType, IdxT>);
  // Manually pack parameters to bypass MSVC variadic template deduction issues
  IdxT size_param = static_cast<IdxT>(size);
  auto shape_param = const_param(shape);
  auto strides_in_param = const_param(strides_in);
  auto strides_out_param = const_param(strides_out);
  void* params[] = {
      (void*)&in_ptr,
      (void*)&out_ptr,
      (void*)&size_param,
      (void*)&shape_param,
      (void*)&strides_in_param,
      (void*)&strides_out_param,
      (void*)&ndim,
      (void*)&offset_in,
      (void*)&offset_out};
  encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
}

void copy_general_dynamic(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const array& dynamic_offset_in,
    const array& dynamic_offset_out) {
#ifdef _MSC_VER
  // On Windows, use runtime-selected registered kernel to work around MSVC
  // template issues.
  bool large = in.data_size() > INT32_MAX || out.data_size() > INT32_MAX;

  // Only handle same-type copies for now (most common case for
  // DynamicSliceUpdate)
  if (in.dtype() == out.dtype()) {
    int ndim = shape.size();
    auto [num_blocks, block_dims] = get_launch_args(out, large);

    void* kernel = nullptr;
    if (ndim <= 3) {
      kernel = get_copy_gg_dynamic_nd_kernel_by_dtype(in.dtype(), ndim, large);
    } else {
      kernel = get_copy_gg_dynamic_kernel_by_dtype(in.dtype(), large);
    }

    if (kernel != nullptr) {
      // Get raw GPU pointers - must use gpu_ptr, not data<void>() which may
      // copy to managed/host memory
      const void* in_ptr = static_cast<const char*>(gpu_ptr<void>(in)) +
          offset_in * size_of(in.dtype());
      void* out_ptr = static_cast<char*>(gpu_ptr<void>(out)) +
          offset_out * size_of(out.dtype());
      int32_t size_param = static_cast<int32_t>(out.size());
      const int64_t* dyn_off_in = gpu_ptr<int64_t>(dynamic_offset_in);
      const int64_t* dyn_off_out = gpu_ptr<int64_t>(dynamic_offset_out);

      if (ndim <= 3) {
        // copy_gg_dynamic_nd kernel parameters
        if (ndim == 1) {
          auto shape_1 = const_param<1>(shape);
          auto strides_in_1 = const_param<1>(strides_in);
          auto strides_out_1 = const_param<1>(strides_out);
          void* params[] = {
              (void*)&in_ptr,
              (void*)&out_ptr,
              (void*)&size_param,
              (void*)&shape_1,
              (void*)&strides_in_1,
              (void*)&strides_out_1,
              (void*)&dyn_off_in,
              (void*)&dyn_off_out};
          encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
        } else if (ndim == 2) {
          auto shape_2 = const_param<2>(shape);
          auto strides_in_2 = const_param<2>(strides_in);
          auto strides_out_2 = const_param<2>(strides_out);
          void* params[] = {
              (void*)&in_ptr,
              (void*)&out_ptr,
              (void*)&size_param,
              (void*)&shape_2,
              (void*)&strides_in_2,
              (void*)&strides_out_2,
              (void*)&dyn_off_in,
              (void*)&dyn_off_out};
          encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
        } else { // ndim == 3
          auto shape_3 = const_param<3>(shape);
          auto strides_in_3 = const_param<3>(strides_in);
          auto strides_out_3 = const_param<3>(strides_out);
          void* params[] = {
              (void*)&in_ptr,
              (void*)&out_ptr,
              (void*)&size_param,
              (void*)&shape_3,
              (void*)&strides_in_3,
              (void*)&strides_out_3,
              (void*)&dyn_off_in,
              (void*)&dyn_off_out};
          encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
        }
      } else {
        // copy_gg_dynamic kernel parameters (for ndim >= 4)
        auto shape_param = const_param(shape);
        auto strides_in_param = const_param(strides_in);
        auto strides_out_param = const_param(strides_out);
        void* params[] = {
            (void*)&in_ptr,
            (void*)&out_ptr,
            (void*)&size_param,
            (void*)&shape_param,
            (void*)&strides_in_param,
            (void*)&strides_out_param,
            (void*)&ndim,
            (void*)&dyn_off_in,
            (void*)&dyn_off_out};
        encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
      }
      return;
    }
  }
  // Fall through to template dispatch for unsupported type combinations
#endif

  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(
          in.data_size() > INT32_MAX || out.data_size() > INT32_MAX,
          [&](auto large) {
            using InType = cuda_type_t<MLX_GET_TYPE(in_type_tag)>;
            using OutType = cuda_type_t<MLX_GET_TYPE(out_type_tag)>;
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;
            const InType* in_ptr = gpu_ptr<InType>(in) + offset_in;
            OutType* out_ptr = gpu_ptr<OutType>(out) + offset_out;
            int ndim = shape.size();
            auto [num_blocks, block_dims] = get_launch_args(out, large());
            if (ndim <= 3) {
              dispatch_copy_gg_dynamic_nd_ndim<InType, OutType, IdxT>(
                  encoder,
                  in_ptr,
                  out_ptr,
                  out.size(),
                  ndim,
                  shape,
                  strides_in,
                  strides_out,
                  gpu_ptr<int64_t>(dynamic_offset_in),
                  gpu_ptr<int64_t>(dynamic_offset_out),
                  num_blocks,
                  block_dims);
            } else { // ndim >= 4
              launch_copy_gg_dynamic_kernel<InType, OutType, IdxT>(
                  encoder,
                  in_ptr,
                  out_ptr,
                  out.size(),
                  shape,
                  strides_in,
                  strides_out,
                  ndim,
                  gpu_ptr<int64_t>(dynamic_offset_in),
                  gpu_ptr<int64_t>(dynamic_offset_out),
                  num_blocks,
                  block_dims);
            }
          });
    });
  });
}

} // namespace mlx::core
