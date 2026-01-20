// Copyright © 2025 Apple Inc.
#include "mlx/backend/common/ternary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/ternary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename T, typename IdxT, int N_READS>
__global__ void
ternary_v(const bool* a, const T* b, const T* c, T* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i], c[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);
    auto c_vec = load_vector<N_READS>(c, index);

    AlignedVector<T, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b_vec[i], c_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename T, typename IdxT, int NDIM, int N_READS>
__global__ void ternary_g_nd(
    const bool* a,
    const T* b,
    const T* c,
    T* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> c_strides) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[NDIM - 1];
  auto a_stride_x = a_strides[NDIM - 1];
  auto b_stride_x = b_strides[NDIM - 1];
  auto c_stride_x = c_strides[NDIM - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx, c_idx] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      c_strides.data());
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, false);
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, T(0));
  auto c_vec =
      load_vector<N_READS>(c + c_idx, index_x, shape_x, c_stride_x, T(0));

  AlignedVector<T, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i], c_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename T, typename IdxT, int N_READS>
__global__ void ternary_g(
    const bool* a,
    const T* b,
    const T* c,
    T* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
    const __grid_constant__ Strides c_strides,
    int ndim) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto a_stride_x = a_strides[ndim - 1];
  auto b_stride_x = b_strides[ndim - 1];
  auto c_stride_x = c_strides[ndim - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx, c_idx] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      c_strides.data(),
      ndim);
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, false);
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, T(0));
  auto c_vec =
      load_vector<N_READS>(c + c_idx, index_x, shape_x, c_stride_x, T(0));

  AlignedVector<T, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i], c_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

} // namespace cu

#ifdef _MSC_VER
// Windows: Force kernel instantiation via global volatile pointers
// ternary_v kernels - vectorized (VectorVectorVector case)
volatile void* g_ternary_v_select_float_u32_4 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, float, uint32_t, 4>);
volatile void* g_ternary_v_select_i32_u32_4 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, int32_t, uint32_t, 4>);
volatile void* g_ternary_v_select_bool_u32_16 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, bool, uint32_t, 16>);
volatile void* g_ternary_v_select_u8_u32_16 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, uint8_t, uint32_t, 16>);
volatile void* g_ternary_v_select_i8_u32_16 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, int8_t, uint32_t, 16>);
volatile void* g_ternary_v_select_u16_u32_8 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, uint16_t, uint32_t, 8>);
volatile void* g_ternary_v_select_i16_u32_8 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, int16_t, uint32_t, 8>);
volatile void* g_ternary_v_select_u32_u32_4 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, uint32_t, uint32_t, 4>);
volatile void* g_ternary_v_select_i64_u32_2 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, int64_t, uint32_t, 2>);
volatile void* g_ternary_v_select_u64_u32_2 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, uint64_t, uint32_t, 2>);
volatile void* g_ternary_v_select_f16_u32_8 =
    reinterpret_cast<void*>(&cu::ternary_v<cu::Select, half, uint32_t, 8>);
volatile void* g_ternary_v_select_bf16_u32_8 = reinterpret_cast<void*>(
    &cu::ternary_v<cu::Select, bfloat16_t, uint32_t, 8>);
// complex64 support for ternary operations (use cu::complex64_t for CUDA device
// code)
volatile void* g_ternary_v_select_c64_u32_2 = reinterpret_cast<void*>(
    &cu::ternary_v<cu::Select, cu::complex64_t, uint32_t, 2>);

// ternary_g_nd kernels - general N-dimensional (ndim <= 3)
volatile void* g_ternary_g_nd_select_float_i32_1_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, float, int32_t, 1, 1>);
volatile void* g_ternary_g_nd_select_float_i32_1_4 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, float, int32_t, 1, 4>);
volatile void* g_ternary_g_nd_select_float_i32_2_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, float, int32_t, 2, 1>);
volatile void* g_ternary_g_nd_select_float_i32_2_4 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, float, int32_t, 2, 4>);
volatile void* g_ternary_g_nd_select_float_i32_3_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, float, int32_t, 3, 1>);
volatile void* g_ternary_g_nd_select_float_i32_3_4 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, float, int32_t, 3, 4>);
volatile void* g_ternary_g_nd_select_i32_i32_1_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, int32_t, int32_t, 1, 1>);
volatile void* g_ternary_g_nd_select_i32_i32_1_4 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, int32_t, int32_t, 1, 4>);
volatile void* g_ternary_g_nd_select_i32_i32_2_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, int32_t, int32_t, 2, 1>);
volatile void* g_ternary_g_nd_select_i32_i32_2_4 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, int32_t, int32_t, 2, 4>);
volatile void* g_ternary_g_nd_select_i32_i32_3_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, int32_t, int32_t, 3, 1>);
volatile void* g_ternary_g_nd_select_i32_i32_3_4 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, int32_t, int32_t, 3, 4>);
volatile void* g_ternary_g_nd_select_bool_i32_1_1 =
    reinterpret_cast<void*>(&cu::ternary_g_nd<cu::Select, bool, int32_t, 1, 1>);
volatile void* g_ternary_g_nd_select_bool_i32_1_4 =
    reinterpret_cast<void*>(&cu::ternary_g_nd<cu::Select, bool, int32_t, 1, 4>);
volatile void* g_ternary_g_nd_select_bool_i32_2_1 =
    reinterpret_cast<void*>(&cu::ternary_g_nd<cu::Select, bool, int32_t, 2, 1>);
volatile void* g_ternary_g_nd_select_bool_i32_2_4 =
    reinterpret_cast<void*>(&cu::ternary_g_nd<cu::Select, bool, int32_t, 2, 4>);
volatile void* g_ternary_g_nd_select_bool_i32_3_1 =
    reinterpret_cast<void*>(&cu::ternary_g_nd<cu::Select, bool, int32_t, 3, 1>);
volatile void* g_ternary_g_nd_select_bool_i32_3_4 =
    reinterpret_cast<void*>(&cu::ternary_g_nd<cu::Select, bool, int32_t, 3, 4>);
// complex64 support for general N-dimensional ternary operations
volatile void* g_ternary_g_nd_select_c64_i32_1_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, cu::complex64_t, int32_t, 1, 1>);
volatile void* g_ternary_g_nd_select_c64_i32_1_2 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, cu::complex64_t, int32_t, 1, 2>);
volatile void* g_ternary_g_nd_select_c64_i32_2_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, cu::complex64_t, int32_t, 2, 1>);
volatile void* g_ternary_g_nd_select_c64_i32_2_2 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, cu::complex64_t, int32_t, 2, 2>);
volatile void* g_ternary_g_nd_select_c64_i32_3_1 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, cu::complex64_t, int32_t, 3, 1>);
volatile void* g_ternary_g_nd_select_c64_i32_3_2 = reinterpret_cast<void*>(
    &cu::ternary_g_nd<cu::Select, cu::complex64_t, int32_t, 3, 2>);

// ternary_g kernels - general (ndim > 3)
volatile void* g_ternary_g_select_float_i32_1 =
    reinterpret_cast<void*>(&cu::ternary_g<cu::Select, float, int32_t, 1>);
volatile void* g_ternary_g_select_float_i32_4 =
    reinterpret_cast<void*>(&cu::ternary_g<cu::Select, float, int32_t, 4>);
volatile void* g_ternary_g_select_i32_i32_1 =
    reinterpret_cast<void*>(&cu::ternary_g<cu::Select, int32_t, int32_t, 1>);
volatile void* g_ternary_g_select_i32_i32_4 =
    reinterpret_cast<void*>(&cu::ternary_g<cu::Select, int32_t, int32_t, 4>);
volatile void* g_ternary_g_select_bool_i32_1 =
    reinterpret_cast<void*>(&cu::ternary_g<cu::Select, bool, int32_t, 1>);
volatile void* g_ternary_g_select_bool_i32_4 =
    reinterpret_cast<void*>(&cu::ternary_g<cu::Select, bool, int32_t, 4>);
// complex64 support for general ternary operations (ndim > 3)
volatile void* g_ternary_g_select_c64_i32_1 = reinterpret_cast<void*>(
    &cu::ternary_g<cu::Select, cu::complex64_t, int32_t, 1>);
volatile void* g_ternary_g_select_c64_i32_2 = reinterpret_cast<void*>(
    &cu::ternary_g<cu::Select, cu::complex64_t, int32_t, 2>);

// Runtime kernel selector functions
void* get_ternary_v_kernel(Dtype dtype, bool large) {
  if (large)
    return nullptr;
  switch (dtype.val()) {
    case float32:
      return const_cast<void*>(g_ternary_v_select_float_u32_4);
    case int32:
      return const_cast<void*>(g_ternary_v_select_i32_u32_4);
    case bool_:
      return const_cast<void*>(g_ternary_v_select_bool_u32_16);
    case uint8:
      return const_cast<void*>(g_ternary_v_select_u8_u32_16);
    case int8:
      return const_cast<void*>(g_ternary_v_select_i8_u32_16);
    case uint16:
      return const_cast<void*>(g_ternary_v_select_u16_u32_8);
    case int16:
      return const_cast<void*>(g_ternary_v_select_i16_u32_8);
    case uint32:
      return const_cast<void*>(g_ternary_v_select_u32_u32_4);
    case int64:
      return const_cast<void*>(g_ternary_v_select_i64_u32_2);
    case uint64:
      return const_cast<void*>(g_ternary_v_select_u64_u32_2);
    case float16:
      return const_cast<void*>(g_ternary_v_select_f16_u32_8);
    case bfloat16:
      return const_cast<void*>(g_ternary_v_select_bf16_u32_8);
    case complex64:
      return const_cast<void*>(g_ternary_v_select_c64_u32_2);
    default:
      return nullptr;
  }
}

void* get_ternary_g_nd_kernel(Dtype dtype, int ndim, int n_reads, bool large) {
  if (large)
    return nullptr;
  if (ndim == 1) {
    switch (dtype.val()) {
      case float32:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_float_i32_1_4)
            : const_cast<void*>(g_ternary_g_nd_select_float_i32_1_1);
      case int32:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_i32_i32_1_4)
            : const_cast<void*>(g_ternary_g_nd_select_i32_i32_1_1);
      case bool_:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_bool_i32_1_4)
            : const_cast<void*>(g_ternary_g_nd_select_bool_i32_1_1);
      case complex64:
        // complex64 is 8 bytes, so N_READS=2 for 16-byte reads
        return n_reads >= 2
            ? const_cast<void*>(g_ternary_g_nd_select_c64_i32_1_2)
            : const_cast<void*>(g_ternary_g_nd_select_c64_i32_1_1);
      default:
        return nullptr;
    }
  } else if (ndim == 2) {
    switch (dtype.val()) {
      case float32:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_float_i32_2_4)
            : const_cast<void*>(g_ternary_g_nd_select_float_i32_2_1);
      case int32:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_i32_i32_2_4)
            : const_cast<void*>(g_ternary_g_nd_select_i32_i32_2_1);
      case bool_:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_bool_i32_2_4)
            : const_cast<void*>(g_ternary_g_nd_select_bool_i32_2_1);
      case complex64:
        return n_reads >= 2
            ? const_cast<void*>(g_ternary_g_nd_select_c64_i32_2_2)
            : const_cast<void*>(g_ternary_g_nd_select_c64_i32_2_1);
      default:
        return nullptr;
    }
  } else if (ndim == 3) {
    switch (dtype.val()) {
      case float32:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_float_i32_3_4)
            : const_cast<void*>(g_ternary_g_nd_select_float_i32_3_1);
      case int32:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_i32_i32_3_4)
            : const_cast<void*>(g_ternary_g_nd_select_i32_i32_3_1);
      case bool_:
        return n_reads == 4
            ? const_cast<void*>(g_ternary_g_nd_select_bool_i32_3_4)
            : const_cast<void*>(g_ternary_g_nd_select_bool_i32_3_1);
      case complex64:
        return n_reads >= 2
            ? const_cast<void*>(g_ternary_g_nd_select_c64_i32_3_2)
            : const_cast<void*>(g_ternary_g_nd_select_c64_i32_3_1);
      default:
        return nullptr;
    }
  }
  return nullptr;
}

void* get_ternary_g_kernel(Dtype dtype, int n_reads, bool large) {
  if (large)
    return nullptr;
  switch (dtype.val()) {
    case float32:
      return n_reads == 4 ? const_cast<void*>(g_ternary_g_select_float_i32_4)
                          : const_cast<void*>(g_ternary_g_select_float_i32_1);
    case int32:
      return n_reads == 4 ? const_cast<void*>(g_ternary_g_select_i32_i32_4)
                          : const_cast<void*>(g_ternary_g_select_i32_i32_1);
    case bool_:
      return n_reads == 4 ? const_cast<void*>(g_ternary_g_select_bool_i32_4)
                          : const_cast<void*>(g_ternary_g_select_bool_i32_1);
    case complex64:
      return n_reads >= 2 ? const_cast<void*>(g_ternary_g_select_c64_i32_2)
                          : const_cast<void*>(g_ternary_g_select_c64_i32_1);
    default:
      return nullptr;
  }
}
#endif

template <typename Op>
void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  const auto& c = inputs[2];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);
  dispatch_all_types(out.dtype(), [&](auto type_tag) {
    using DType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    auto topt = get_ternary_op_type(a, b, c);
    if (topt == TernaryOpType::VectorVectorVector ||
        topt == TernaryOpType::ScalarScalarScalar) {
#ifdef _MSC_VER
      // Windows: Use runtime kernel selection
      bool large = out.data_size() > UINT32_MAX;
      void* kernel = get_ternary_v_kernel(out.dtype(), large);
      if (kernel != nullptr) {
        constexpr int N_READS = 16 / sizeof(DType);
        auto [num_blocks, block_dims] = get_launch_args(
            out.data_size(), out.shape(), out.strides(), large, N_READS);
        const bool* a_ptr = gpu_ptr<bool>(a);
        const DType* b_ptr = gpu_ptr<DType>(b);
        const DType* c_ptr = gpu_ptr<DType>(c);
        DType* out_ptr = gpu_ptr<DType>(out);
        uint32_t size = out.data_size();
        void* params[] = {&a_ptr, &b_ptr, &c_ptr, &out_ptr, &size};
        encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
      } else
#endif
      {
        dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
          using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
          constexpr int N_READS = 16 / sizeof(DType);
          auto [num_blocks, block_dims] = get_launch_args(
              out.data_size(), out.shape(), out.strides(), large(), N_READS);
          auto kernel = cu::ternary_v<Op, DType, IdxT, N_READS>;
          // Store params in variables to ensure they remain valid
          const bool* a_ptr = gpu_ptr<bool>(a);
          const DType* b_ptr = gpu_ptr<DType>(b);
          const DType* c_ptr = gpu_ptr<DType>(c);
          DType* out_ptr = gpu_ptr<DType>(out);
          IdxT size = out.data_size();
          void* params[] = {&a_ptr, &b_ptr, &c_ptr, &out_ptr, &size};
          encoder.add_kernel_node(
              reinterpret_cast<void*>(kernel),
              num_blocks,
              block_dims,
              0,
              params);
        });
      }
    } else {
      dispatch_bool(
          a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
              c.data_size() > INT32_MAX || out.data_size() > INT32_MAX,
          [&](auto large) {
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;
            Shape shape;
            std::vector<Strides> strides;
            std::tie(shape, strides) = collapse_contiguous_dims(a, b, c, out);
            auto& a_strides = strides[0];
            auto& b_strides = strides[1];
            auto& c_strides = strides[2];
            int ndim = shape.size();
            // N_READS should be 16 / sizeof(DType) for optimal reads
            // For complex64 (8 bytes), this is 2, not 4
            constexpr int N_READS = 16 / sizeof(DType);
            int work_per_thread = 1;
            auto dim0 = ndim > 0 ? shape.back() : 1;
            auto rest = out.size() / dim0;
            if (dim0 >= N_READS) {
              work_per_thread = N_READS;
            }
            dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
            auto block_dims = get_block_dims(dim0, rest, 1);
            uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
            uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);

            if (ndim <= 3) {
#ifdef _MSC_VER
              // Windows: Use runtime kernel selection
              bool large_val = large();
              void* kernel = get_ternary_g_nd_kernel(
                  out.dtype(), ndim, work_per_thread, large_val);
              if (kernel != nullptr) {
                const bool* a_ptr = gpu_ptr<bool>(a);
                const DType* b_ptr = gpu_ptr<DType>(b);
                const DType* c_ptr = gpu_ptr<DType>(c);
                DType* out_ptr = gpu_ptr<DType>(out);
                int32_t rest_val = rest;
                if (ndim == 1) {
                  auto shape_param = const_param<1>(shape);
                  auto a_strides_param = const_param<1>(a_strides);
                  auto b_strides_param = const_param<1>(b_strides);
                  auto c_strides_param = const_param<1>(c_strides);
                  void* params[] = {
                      &a_ptr,
                      &b_ptr,
                      &c_ptr,
                      &out_ptr,
                      &rest_val,
                      &shape_param,
                      &a_strides_param,
                      &b_strides_param,
                      &c_strides_param};
                  encoder.add_kernel_node(
                      kernel,
                      {num_blocks_x, num_blocks_y},
                      block_dims,
                      0,
                      params);
                } else if (ndim == 2) {
                  auto shape_param = const_param<2>(shape);
                  auto a_strides_param = const_param<2>(a_strides);
                  auto b_strides_param = const_param<2>(b_strides);
                  auto c_strides_param = const_param<2>(c_strides);
                  void* params[] = {
                      &a_ptr,
                      &b_ptr,
                      &c_ptr,
                      &out_ptr,
                      &rest_val,
                      &shape_param,
                      &a_strides_param,
                      &b_strides_param,
                      &c_strides_param};
                  encoder.add_kernel_node(
                      kernel,
                      {num_blocks_x, num_blocks_y},
                      block_dims,
                      0,
                      params);
                } else {
                  auto shape_param = const_param<3>(shape);
                  auto a_strides_param = const_param<3>(a_strides);
                  auto b_strides_param = const_param<3>(b_strides);
                  auto c_strides_param = const_param<3>(c_strides);
                  void* params[] = {
                      &a_ptr,
                      &b_ptr,
                      &c_ptr,
                      &out_ptr,
                      &rest_val,
                      &shape_param,
                      &a_strides_param,
                      &b_strides_param,
                      &c_strides_param};
                  encoder.add_kernel_node(
                      kernel,
                      {num_blocks_x, num_blocks_y},
                      block_dims,
                      0,
                      params);
                }
              } else
#endif
              {
                dispatch_1_2_3(ndim, [&](auto dims_constant) {
                  auto kernel =
                      cu::ternary_g_nd<Op, DType, IdxT, dims_constant(), 1>;
                  if (work_per_thread == 4) {
                    kernel =
                        cu::ternary_g_nd<Op, DType, IdxT, dims_constant(), 4>;
                  }
                  // Store params in variables to ensure they remain valid
                  const bool* a_ptr = gpu_ptr<bool>(a);
                  const DType* b_ptr = gpu_ptr<DType>(b);
                  const DType* c_ptr = gpu_ptr<DType>(c);
                  DType* out_ptr = gpu_ptr<DType>(out);
                  IdxT rest_val = rest;
                  auto shape_param = const_param<dims_constant()>(shape);
                  auto a_strides_param =
                      const_param<dims_constant()>(a_strides);
                  auto b_strides_param =
                      const_param<dims_constant()>(b_strides);
                  auto c_strides_param =
                      const_param<dims_constant()>(c_strides);
                  void* params[] = {
                      &a_ptr,
                      &b_ptr,
                      &c_ptr,
                      &out_ptr,
                      &rest_val,
                      &shape_param,
                      &a_strides_param,
                      &b_strides_param,
                      &c_strides_param};
                  encoder.add_kernel_node(
                      reinterpret_cast<void*>(kernel),
                      {num_blocks_x, num_blocks_y},
                      block_dims,
                      0,
                      params);
                });
              }
            } else {
#ifdef _MSC_VER
              // Windows: Use runtime kernel selection
              bool large_val = large();
              void* kernel =
                  get_ternary_g_kernel(out.dtype(), work_per_thread, large_val);
              if (kernel != nullptr) {
                const bool* a_ptr = gpu_ptr<bool>(a);
                const DType* b_ptr = gpu_ptr<DType>(b);
                const DType* c_ptr = gpu_ptr<DType>(c);
                DType* out_ptr = gpu_ptr<DType>(out);
                int32_t rest_val = rest;
                auto shape_param = const_param(shape);
                auto a_strides_param = const_param(a_strides);
                auto b_strides_param = const_param(b_strides);
                auto c_strides_param = const_param(c_strides);
                int ndim_val = ndim;
                void* params[] = {
                    &a_ptr,
                    &b_ptr,
                    &c_ptr,
                    &out_ptr,
                    &rest_val,
                    &shape_param,
                    &a_strides_param,
                    &b_strides_param,
                    &c_strides_param,
                    &ndim_val};
                encoder.add_kernel_node(
                    kernel,
                    {num_blocks_x, num_blocks_y},
                    block_dims,
                    0,
                    params);
              } else
#endif
              {
                auto kernel = cu::ternary_g<Op, DType, IdxT, 1>;
                if (work_per_thread == 4) {
                  kernel = cu::ternary_g<Op, DType, IdxT, 4>;
                }
                // Store params in variables to ensure they remain valid
                const bool* a_ptr = gpu_ptr<bool>(a);
                const DType* b_ptr = gpu_ptr<DType>(b);
                const DType* c_ptr = gpu_ptr<DType>(c);
                DType* out_ptr = gpu_ptr<DType>(out);
                IdxT rest_val = rest;
                auto shape_param = const_param(shape);
                auto a_strides_param = const_param(a_strides);
                auto b_strides_param = const_param(b_strides);
                auto c_strides_param = const_param(c_strides);
                int ndim_val = ndim;
                void* params[] = {
                    &a_ptr,
                    &b_ptr,
                    &c_ptr,
                    &out_ptr,
                    &rest_val,
                    &shape_param,
                    &a_strides_param,
                    &b_strides_param,
                    &c_strides_param,
                    &ndim_val};
                encoder.add_kernel_node(
                    reinterpret_cast<void*>(kernel),
                    {num_blocks_x, num_blocks_y},
                    block_dims,
                    0,
                    params);
              }
            }
          });
    }
  });
}

template <typename Op>
void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];
  auto topt = get_ternary_op_type(a, b, c);
  auto& encoder = cu::get_command_encoder(s);
  set_ternary_op_output_data(
      a, b, c, out, topt, [&](auto n) { return cu::malloc_async(n, encoder); });
  ternary_op_gpu_inplace<Op>(inputs, out, s);
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Select::eval_gpu");
  auto& s = out.primitive().stream();
  ternary_op_gpu<cu::Select>(inputs, out, s);
}

} // namespace mlx::core
