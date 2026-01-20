// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"
#include "mlx/backend/cuda/utils.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int NDIM, int N_READS>
__global__ void copy_gg_nd(
    const In* in,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_in,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_out) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[NDIM - 1];
  auto in_stride_x = strides_in[NDIM - 1];
  auto out_stride_x = strides_out[NDIM - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [idx_in, idx_out] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x,
      shape.data(),
      strides_in.data(),
      strides_out.data());

  auto in_vec =
      load_vector<N_READS>(in + idx_in, index_x, shape_x, in_stride_x, In(0));
  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = CastOp<In, Out>{}(in_vec[i]);
  }
  store_vector(out + idx_out, index_x, out_vec, shape_x, out_stride_x);
}

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_gg(
    const In* in,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides strides_in,
    const __grid_constant__ Strides strides_out,
    int ndim) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto in_stride_x = strides_in[ndim - 1];
  auto out_stride_x = strides_out[ndim - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [idx_in, idx_out] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      strides_in.data(),
      strides_out.data(),
      ndim);

  auto in_vec =
      load_vector<N_READS>(in + idx_in, index_x, shape_x, in_stride_x, In(0));
  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = CastOp<In, Out>{}(in_vec[i]);
  }
  store_vector(out + idx_out, index_x, out_vec, shape_x, out_stride_x);
}

} // namespace cu

// ============================================================================
// Macros for kernel declarations - DRY pattern for Windows/MSVC registration
// ============================================================================
// On Windows, MSVC doesn't properly resolve template types through nested
// lambdas with dispatch_all_types. These macros generate global volatile
// pointers that force kernel instantiation and registration at static init
// time.

// Macro to declare copy_gg_nd kernel pointers for all 3 NDIMs with given
// N_READS
#define DECL_COPY_GG_ND_ALL_NDIM(InT, OutT, name, N_READS)                    \
  volatile void* g_copy_gg_nd_##name##_1_##N_READS = reinterpret_cast<void*>( \
      &cu::copy_gg_nd<InT, OutT, int32_t, 1, N_READS>);                       \
  volatile void* g_copy_gg_nd_##name##_1_1 =                                  \
      reinterpret_cast<void*>(&cu::copy_gg_nd<InT, OutT, int32_t, 1, 1>);     \
  volatile void* g_copy_gg_nd_##name##_2_##N_READS = reinterpret_cast<void*>( \
      &cu::copy_gg_nd<InT, OutT, int32_t, 2, N_READS>);                       \
  volatile void* g_copy_gg_nd_##name##_2_1 =                                  \
      reinterpret_cast<void*>(&cu::copy_gg_nd<InT, OutT, int32_t, 2, 1>);     \
  volatile void* g_copy_gg_nd_##name##_3_##N_READS = reinterpret_cast<void*>( \
      &cu::copy_gg_nd<InT, OutT, int32_t, 3, N_READS>);                       \
  volatile void* g_copy_gg_nd_##name##_3_1 =                                  \
      reinterpret_cast<void*>(&cu::copy_gg_nd<InT, OutT, int32_t, 3, 1>);

// Macro to declare copy_gg kernel pointers (for ndim >= 4)
#define DECL_COPY_GG(InT, OutT, name, N_READS)                            \
  volatile void* g_copy_gg_##name##_##N_READS =                           \
      reinterpret_cast<void*>(&cu::copy_gg<InT, OutT, int32_t, N_READS>); \
  volatile void* g_copy_gg_##name##_1 =                                   \
      reinterpret_cast<void*>(&cu::copy_gg<InT, OutT, int32_t, 1>);

// Macro for dispatching copy_gg_nd by dtype and ndim
#define DISPATCH_COPY_GG_ND(dtype_val, name, N_READS)            \
  case dtype_val:                                                \
    if (ndim == 1)                                               \
      return n_reads >= N_READS                                  \
          ? const_cast<void*>(g_copy_gg_nd_##name##_1_##N_READS) \
          : const_cast<void*>(g_copy_gg_nd_##name##_1_1);        \
    if (ndim == 2)                                               \
      return n_reads >= N_READS                                  \
          ? const_cast<void*>(g_copy_gg_nd_##name##_2_##N_READS) \
          : const_cast<void*>(g_copy_gg_nd_##name##_2_1);        \
    if (ndim == 3)                                               \
      return n_reads >= N_READS                                  \
          ? const_cast<void*>(g_copy_gg_nd_##name##_3_##N_READS) \
          : const_cast<void*>(g_copy_gg_nd_##name##_3_1);        \
    break;

// Macro for dispatching copy_gg by dtype (ndim >= 4)
#define DISPATCH_COPY_GG(dtype_val, name, N_READS)        \
  case dtype_val:                                         \
    return n_reads >= N_READS                             \
        ? const_cast<void*>(g_copy_gg_##name##_##N_READS) \
        : const_cast<void*>(g_copy_gg_##name##_1);

// ============================================================================
// Kernel declarations using macros
// ============================================================================
// N_READS based on type size: 1->16, 2->8, 4->4, 8->2

// Same-type copies for copy_gg_nd (all ndim 1,2,3)
DECL_COPY_GG_ND_ALL_NDIM(float, float, float, 4)
DECL_COPY_GG_ND_ALL_NDIM(int32_t, int32_t, i32, 4)
DECL_COPY_GG_ND_ALL_NDIM(uint32_t, uint32_t, u32, 4)
DECL_COPY_GG_ND_ALL_NDIM(bool, bool, bool, 4)
DECL_COPY_GG_ND_ALL_NDIM(__half, __half, half, 4)
DECL_COPY_GG_ND_ALL_NDIM(__nv_bfloat16, __nv_bfloat16, bf16, 4)
DECL_COPY_GG_ND_ALL_NDIM(int64_t, int64_t, i64, 2)
DECL_COPY_GG_ND_ALL_NDIM(uint64_t, uint64_t, u64, 2)
DECL_COPY_GG_ND_ALL_NDIM(double, double, double, 2)
DECL_COPY_GG_ND_ALL_NDIM(cu::complex64_t, cu::complex64_t, c64, 4)
DECL_COPY_GG_ND_ALL_NDIM(int8_t, int8_t, i8, 16)
DECL_COPY_GG_ND_ALL_NDIM(uint8_t, uint8_t, u8, 16)
DECL_COPY_GG_ND_ALL_NDIM(int16_t, int16_t, i16, 8)
DECL_COPY_GG_ND_ALL_NDIM(uint16_t, uint16_t, u16, 8)

// Cross-type copies for complex64 conversions
DECL_COPY_GG_ND_ALL_NDIM(float, cu::complex64_t, float_c64, 4)
DECL_COPY_GG_ND_ALL_NDIM(cu::complex64_t, float, c64_float, 2)
DECL_COPY_GG_ND_ALL_NDIM(bool, cu::complex64_t, bool_c64, 4)
DECL_COPY_GG_ND_ALL_NDIM(cu::complex64_t, bool, c64_bool, 2)

// Same-type copies for copy_gg (ndim >= 4)
DECL_COPY_GG(float, float, float, 4)
DECL_COPY_GG(int32_t, int32_t, i32, 4)
DECL_COPY_GG(bool, bool, bool, 4)
DECL_COPY_GG(cu::complex64_t, cu::complex64_t, c64, 4)
DECL_COPY_GG(__half, __half, half, 4)
DECL_COPY_GG(__nv_bfloat16, __nv_bfloat16, bf16, 4)

// int64 index variants for large arrays
volatile void* g_copy_gg_nd_float_2_4_i64 =
    reinterpret_cast<void*>(&cu::copy_gg_nd<float, float, int64_t, 2, 4>);
volatile void* g_copy_gg_float_4_i64 =
    reinterpret_cast<void*>(&cu::copy_gg<float, float, int64_t, 4>);

// ============================================================================
// Runtime kernel selector for Windows/MSVC
// ============================================================================
#ifdef _MSC_VER
#include "mlx/dtype.h"

// Get the registered kernel for copy_gg_nd based on runtime dtype and ndim
void* get_copy_gg_nd_kernel_by_dtype(
    Dtype dtype,
    int ndim,
    int work_per_thread,
    bool large) {
  if (large)
    return nullptr;

  int n_reads = (work_per_thread == 4) ? 4 : 1;

  switch (dtype.val()) {
    DISPATCH_COPY_GG_ND(float32, float, 4)
    DISPATCH_COPY_GG_ND(int32, i32, 4)
    DISPATCH_COPY_GG_ND(uint32, u32, 4)
    DISPATCH_COPY_GG_ND(bool_, bool, 4)
    DISPATCH_COPY_GG_ND(float16, half, 4)
    DISPATCH_COPY_GG_ND(bfloat16, bf16, 4)
    DISPATCH_COPY_GG_ND(complex64, c64, 4)
    DISPATCH_COPY_GG_ND(int64, i64, 2)
    DISPATCH_COPY_GG_ND(uint64, u64, 2)
    DISPATCH_COPY_GG_ND(float64, double, 2)
    DISPATCH_COPY_GG_ND(int8, i8, 16)
    DISPATCH_COPY_GG_ND(uint8, u8, 16)
    DISPATCH_COPY_GG_ND(int16, i16, 8)
    DISPATCH_COPY_GG_ND(uint16, u16, 8)
    default:
      break;
  }
  return nullptr;
}

// Get the registered kernel for copy_gg based on runtime dtype
void* get_copy_gg_kernel_by_dtype(
    Dtype dtype,
    int work_per_thread,
    bool large) {
  if (large)
    return nullptr;

  int n_reads = (work_per_thread == 4) ? 4 : 1;

  switch (dtype.val()) {
    DISPATCH_COPY_GG(float32, float, 4)
    DISPATCH_COPY_GG(int32, i32, 4)
    DISPATCH_COPY_GG(bool_, bool, 4)
    DISPATCH_COPY_GG(complex64, c64, 4)
    DISPATCH_COPY_GG(float16, half, 4)
    DISPATCH_COPY_GG(bfloat16, bf16, 4)
    default:
      break;
  }
  return nullptr;
}
#endif // _MSC_VER

// Helper template functions to work around MSVC template function pointer
// issues. Cast to void* for MSVC compatibility - avoids template deduction
// issues.
template <
    typename InType,
    typename OutType,
    typename IdxT,
    int NDIM,
    int N_READS>
void launch_copy_gg_nd_kernel(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t rest,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    uint32_t num_blocks_x,
    uint32_t num_blocks_y,
    dim3 block_dims) {
  auto kernel = reinterpret_cast<void*>(
      &cu::copy_gg_nd<InType, OutType, IdxT, NDIM, N_READS>);
  // Manually pack parameters to bypass MSVC variadic template deduction issues
  IdxT rest_param = static_cast<IdxT>(rest);
  auto shape_param = const_param<NDIM>(shape);
  auto strides_in_param = const_param<NDIM>(strides_in);
  auto strides_out_param = const_param<NDIM>(strides_out);
  void* params[] = {
      (void*)&in_ptr,
      (void*)&out_ptr,
      (void*)&rest_param,
      (void*)&shape_param,
      (void*)&strides_in_param,
      (void*)&strides_out_param};
  encoder.add_kernel_node(
      kernel, {num_blocks_x, num_blocks_y}, block_dims, 0, params);
}

template <typename InType, typename OutType, typename IdxT, int NDIM>
void dispatch_copy_gg_nd_n_reads(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t rest,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    uint32_t num_blocks_x,
    uint32_t num_blocks_y,
    dim3 block_dims,
    int work_per_thread) {
  if (work_per_thread == 4) {
    launch_copy_gg_nd_kernel<InType, OutType, IdxT, NDIM, 4>(
        encoder,
        in_ptr,
        out_ptr,
        rest,
        shape,
        strides_in,
        strides_out,
        num_blocks_x,
        num_blocks_y,
        block_dims);
  } else {
    launch_copy_gg_nd_kernel<InType, OutType, IdxT, NDIM, 1>(
        encoder,
        in_ptr,
        out_ptr,
        rest,
        shape,
        strides_in,
        strides_out,
        num_blocks_x,
        num_blocks_y,
        block_dims);
  }
}

template <typename InType, typename OutType, typename IdxT>
void dispatch_copy_gg_nd_ndim(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t rest,
    int ndim,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    uint32_t num_blocks_x,
    uint32_t num_blocks_y,
    dim3 block_dims,
    int work_per_thread) {
  switch (ndim) {
    case 1:
      dispatch_copy_gg_nd_n_reads<InType, OutType, IdxT, 1>(
          encoder,
          in_ptr,
          out_ptr,
          rest,
          shape,
          strides_in,
          strides_out,
          num_blocks_x,
          num_blocks_y,
          block_dims,
          work_per_thread);
      break;
    case 2:
      dispatch_copy_gg_nd_n_reads<InType, OutType, IdxT, 2>(
          encoder,
          in_ptr,
          out_ptr,
          rest,
          shape,
          strides_in,
          strides_out,
          num_blocks_x,
          num_blocks_y,
          block_dims,
          work_per_thread);
      break;
    case 3:
      dispatch_copy_gg_nd_n_reads<InType, OutType, IdxT, 3>(
          encoder,
          in_ptr,
          out_ptr,
          rest,
          shape,
          strides_in,
          strides_out,
          num_blocks_x,
          num_blocks_y,
          block_dims,
          work_per_thread);
      break;
  }
}

template <typename InType, typename OutType, typename IdxT, int N_READS>
void launch_copy_gg_kernel(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t rest,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int ndim,
    uint32_t num_blocks_x,
    uint32_t num_blocks_y,
    dim3 block_dims) {
  // Cast to void* for MSVC compatibility - avoids template deduction issues
  auto kernel =
      reinterpret_cast<void*>(&cu::copy_gg<InType, OutType, IdxT, N_READS>);
  // Manually pack parameters to bypass MSVC variadic template deduction issues
  IdxT rest_param = static_cast<IdxT>(rest);
  auto shape_param = const_param(shape);
  auto strides_in_param = const_param(strides_in);
  auto strides_out_param = const_param(strides_out);
  void* params[] = {
      (void*)&in_ptr,
      (void*)&out_ptr,
      (void*)&rest_param,
      (void*)&shape_param,
      (void*)&strides_in_param,
      (void*)&strides_out_param,
      (void*)&ndim};
  encoder.add_kernel_node(
      kernel, {num_blocks_x, num_blocks_y}, block_dims, 0, params);
}

template <typename InType, typename OutType, typename IdxT>
void dispatch_copy_gg_n_reads(
    cu::CommandEncoder& encoder,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t rest,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int ndim,
    uint32_t num_blocks_x,
    uint32_t num_blocks_y,
    dim3 block_dims,
    int work_per_thread) {
  if (work_per_thread == 4) {
    launch_copy_gg_kernel<InType, OutType, IdxT, 4>(
        encoder,
        in_ptr,
        out_ptr,
        rest,
        shape,
        strides_in,
        strides_out,
        ndim,
        num_blocks_x,
        num_blocks_y,
        block_dims);
  } else {
    launch_copy_gg_kernel<InType, OutType, IdxT, 1>(
        encoder,
        in_ptr,
        out_ptr,
        rest,
        shape,
        strides_in,
        strides_out,
        ndim,
        num_blocks_x,
        num_blocks_y,
        block_dims);
  }
}

void copy_general(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out) {
#ifdef _MSC_VER
  // On Windows, use runtime-selected registered kernel to work around MSVC
  // template issues. MSVC doesn't properly resolve template types through
  // nested lambdas.
  bool large = in.data_size() > INT32_MAX || out.data_size() > INT32_MAX;

  // Only handle same-type copies for now (most common case for SliceUpdate)
  if (in.dtype() == out.dtype()) {
    int ndim = shape.size();
    size_t data_size = 1;
    for (auto& s : shape)
      data_size *= s;

    int work_per_thread = 1;
    auto dim0 = ndim > 0 ? shape.back() : 1;
    auto rest = data_size / dim0;
    if (dim0 >= 4) {
      work_per_thread = 4;
    }

    dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
    auto block_dims = get_block_dims(dim0, rest, 1);
    uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
    uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);

    void* kernel = nullptr;
    if (ndim <= 3) {
      kernel = get_copy_gg_nd_kernel_by_dtype(
          in.dtype(), ndim, work_per_thread, large);
    } else {
      kernel = get_copy_gg_kernel_by_dtype(in.dtype(), work_per_thread, large);
    }

    if (kernel != nullptr) {
      // Get raw GPU pointers - must use gpu_ptr, not data<void>() which may
      // copy to managed/host memory
      const void* in_ptr = static_cast<const char*>(gpu_ptr<void>(in)) +
          offset_in * size_of(in.dtype());
      void* out_ptr = static_cast<char*>(gpu_ptr<void>(out)) +
          offset_out * size_of(out.dtype());
      int32_t rest_param = static_cast<int32_t>(rest);

      if (ndim <= 3) {
        // copy_gg_nd kernel parameters
        auto shape_arr = const_param<3>(shape); // max NDIM=3 for this path
        auto strides_in_arr = const_param<3>(strides_in);
        auto strides_out_arr = const_param<3>(strides_out);

        // Pad arrays for smaller ndim
        if (ndim == 1) {
          auto shape_1 = const_param<1>(shape);
          auto strides_in_1 = const_param<1>(strides_in);
          auto strides_out_1 = const_param<1>(strides_out);
          void* params[] = {
              (void*)&in_ptr,
              (void*)&out_ptr,
              (void*)&rest_param,
              (void*)&shape_1,
              (void*)&strides_in_1,
              (void*)&strides_out_1};
          encoder.add_kernel_node(
              kernel, {num_blocks_x, num_blocks_y}, block_dims, 0, params);
        } else if (ndim == 2) {
          auto shape_2 = const_param<2>(shape);
          auto strides_in_2 = const_param<2>(strides_in);
          auto strides_out_2 = const_param<2>(strides_out);
          void* params[] = {
              (void*)&in_ptr,
              (void*)&out_ptr,
              (void*)&rest_param,
              (void*)&shape_2,
              (void*)&strides_in_2,
              (void*)&strides_out_2};
          encoder.add_kernel_node(
              kernel, {num_blocks_x, num_blocks_y}, block_dims, 0, params);
        } else { // ndim == 3
          void* params[] = {
              (void*)&in_ptr,
              (void*)&out_ptr,
              (void*)&rest_param,
              (void*)&shape_arr,
              (void*)&strides_in_arr,
              (void*)&strides_out_arr};
          encoder.add_kernel_node(
              kernel, {num_blocks_x, num_blocks_y}, block_dims, 0, params);
        }
      } else {
        // copy_gg kernel parameters (for ndim >= 4)
        auto shape_param = const_param(shape);
        auto strides_in_param = const_param(strides_in);
        auto strides_out_param = const_param(strides_out);
        void* params[] = {
            (void*)&in_ptr,
            (void*)&out_ptr,
            (void*)&rest_param,
            (void*)&shape_param,
            (void*)&strides_in_param,
            (void*)&strides_out_param,
            (void*)&ndim};
        encoder.add_kernel_node(
            kernel, {num_blocks_x, num_blocks_y}, block_dims, 0, params);
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
            size_t data_size = 1;
            for (auto& s : shape)
              data_size *= s;

            int work_per_thread = 1;
            auto dim0 = ndim > 0 ? shape.back() : 1;
            auto rest = data_size / dim0;
            if (dim0 >= 4) {
              work_per_thread = 4;
            }

            dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
            auto block_dims = get_block_dims(dim0, rest, 1);
            uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
            uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);

            if (ndim <= 3) {
              dispatch_copy_gg_nd_ndim<InType, OutType, IdxT>(
                  encoder,
                  in_ptr,
                  out_ptr,
                  rest,
                  ndim,
                  shape,
                  strides_in,
                  strides_out,
                  num_blocks_x,
                  num_blocks_y,
                  block_dims,
                  work_per_thread);
            } else { // ndim >= 4
              dispatch_copy_gg_n_reads<InType, OutType, IdxT>(
                  encoder,
                  in_ptr,
                  out_ptr,
                  rest,
                  shape,
                  strides_in,
                  strides_out,
                  ndim,
                  num_blocks_x,
                  num_blocks_y,
                  block_dims,
                  work_per_thread);
            }
          });
    });
  });
}

} // namespace mlx::core
