// Copyright © 2025 Apple Inc.

// Windows-only: All binary operations with INLINE kernel definitions.
//
// CRITICAL: On Windows with NVCC, CUDA kernel templates defined in .cuh headers
// are NOT registered with the CUDA runtime (even with explicit instantiation,
// single TU, /WHOLEARCHIVE, etc.). However, kernels defined DIRECTLY in .cu
// files ARE registered correctly.
//
// This file defines all binary kernels INLINE (not via header include) to
// ensure proper CUDA runtime registration on Windows.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_ss(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (int i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[0], b[0]);
    }
  } else {
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a[0], b[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_sv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[0], b[i]);
    }
  } else {
    auto b_vec = load_vector<N_READS>(b, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a[0], b_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_vs(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[0]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_vv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <
    typename Op,
    typename In,
    typename Out,
    typename IdxT,
    int NDIM,
    int N_READS>
__global__ void binary_g_nd(
    const In* a,
    const In* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides) {
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
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x, shape.data(), a_strides.data(), b_strides.data());
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, In(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, In(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_g(
    const In* a,
    const In* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
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
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      ndim);
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, In(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, In(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_op() {
  if (std::is_same_v<Op, Add> || std::is_same_v<Op, Divide> ||
      std::is_same_v<Op, Maximum> || std::is_same_v<Op, Minimum> ||
      std::is_same_v<Op, Multiply> || std::is_same_v<Op, Subtract> ||
      std::is_same_v<Op, Power> || std::is_same_v<Op, Remainder>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, Equal> || std::is_same_v<Op, Greater> ||
      std::is_same_v<Op, GreaterEqual> || std::is_same_v<Op, Less> ||
      std::is_same_v<Op, LessEqual> || std::is_same_v<Op, NotEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if (std::is_same_v<Op, LogicalAnd> || std::is_same_v<Op, LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, NaNEqual>) {
    return std::is_same_v<Out, bool> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, LogAddExp>) {
    return std::is_same_v<In, Out> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, ArcTan2>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, BitwiseAnd> || std::is_same_v<Op, BitwiseOr> ||
      std::is_same_v<Op, BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  }
  if (std::is_same_v<Op, LeftShift> || std::is_same_v<Op, RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  return false;
}

} // namespace cu

// ============================================================================
// DISPATCH FUNCTIONS
// ============================================================================

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  assert(inputs.size() > 1);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  dispatch_all_types(a.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using CTYPE_IN = MLX_GET_TYPE(in_type_tag);
      using CTYPE_OUT = MLX_GET_TYPE(out_type_tag);
      if constexpr (cu::supports_binary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
        using InType = cuda_type_t<CTYPE_IN>;
        using OutType = cuda_type_t<CTYPE_OUT>;
        auto bopt = get_binary_op_type(a, b);
        if (bopt == BinaryOpType::General) {
          dispatch_bool(
              a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
                  out.data_size() > INT32_MAX,
              [&](auto large) {
                using IdxT = std::conditional_t<large(), int64_t, int32_t>;
                Shape shape;
                std::vector<Strides> strides;
                std::tie(shape, strides) = collapse_contiguous_dims(a, b, out);
                auto& a_strides = strides[0];
                auto& b_strides = strides[1];
                int ndim = shape.size();
                int work_per_thread = 1;
                auto dim0 = ndim > 0 ? shape.back() : 1;
                auto rest = out.size() / dim0;
                if (dim0 >= 4) {
                  work_per_thread = 4;
                }
                dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
                auto block_dims = get_block_dims(dim0, rest, 1);
                uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
                uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);
                if (ndim <= 3) {
                  dispatch_1_2_3(ndim, [&](auto dims_constant) {
                    auto kernel = cu::binary_g_nd<
                        Op,
                        InType,
                        OutType,
                        IdxT,
                        dims_constant(),
                        1>;
                    if (work_per_thread == 4) {
                      kernel = cu::binary_g_nd<
                          Op,
                          InType,
                          OutType,
                          IdxT,
                          dims_constant(),
                          4>;
                    }
                    const InType* a_ptr = gpu_ptr<InType>(a);
                    const InType* b_ptr = gpu_ptr<InType>(b);
                    OutType* out_ptr = gpu_ptr<OutType>(out);
                    IdxT rest_val = rest;
                    auto shape_param = const_param<dims_constant()>(shape);
                    auto a_strides_param =
                        const_param<dims_constant()>(a_strides);
                    auto b_strides_param =
                        const_param<dims_constant()>(b_strides);
                    void* params[] = {
                        &a_ptr,
                        &b_ptr,
                        &out_ptr,
                        &rest_val,
                        &shape_param,
                        &a_strides_param,
                        &b_strides_param};
                    encoder.add_kernel_node(
                        reinterpret_cast<void*>(kernel),
                        {num_blocks_x, num_blocks_y},
                        block_dims,
                        0,
                        params);
                  });
                } else {
                  auto kernel = cu::binary_g<Op, InType, OutType, IdxT, 1>;
                  if (work_per_thread == 4) {
                    kernel = cu::binary_g<Op, InType, OutType, IdxT, 4>;
                  }
                  const InType* a_ptr = gpu_ptr<InType>(a);
                  const InType* b_ptr = gpu_ptr<InType>(b);
                  OutType* out_ptr = gpu_ptr<OutType>(out);
                  IdxT rest_val = rest;
                  auto shape_param = const_param(shape);
                  auto a_strides_param = const_param(a_strides);
                  auto b_strides_param = const_param(b_strides);
                  int ndim_val = ndim;
                  void* params[] = {
                      &a_ptr,
                      &b_ptr,
                      &out_ptr,
                      &rest_val,
                      &shape_param,
                      &a_strides_param,
                      &b_strides_param,
                      &ndim_val};
                  encoder.add_kernel_node(
                      reinterpret_cast<void*>(kernel),
                      {num_blocks_x, num_blocks_y},
                      block_dims,
                      0,
                      params);
                }
              });
        } else {
          dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
            using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
            constexpr int N_READS = 16 / sizeof(InType);
            auto kernel = cu::binary_ss<Op, InType, OutType, IdxT, N_READS>;
            if (bopt == BinaryOpType::ScalarVector) {
              kernel = cu::binary_sv<Op, InType, OutType, IdxT, N_READS>;
            } else if (bopt == BinaryOpType::VectorScalar) {
              kernel = cu::binary_vs<Op, InType, OutType, IdxT, N_READS>;
            } else if (bopt == BinaryOpType::VectorVector) {
              kernel = cu::binary_vv<Op, InType, OutType, IdxT, N_READS>;
            }
            auto [num_blocks, block_dims] = get_launch_args(
                out.data_size(), out.shape(), out.strides(), large(), N_READS);
            const InType* a_ptr = gpu_ptr<InType>(a);
            const InType* b_ptr = gpu_ptr<InType>(b);
            OutType* out_ptr = gpu_ptr<OutType>(out);
            IdxT size = out.data_size();
            void* params[] = {&a_ptr, &b_ptr, &out_ptr, &size};
            encoder.add_kernel_node(
                reinterpret_cast<void*>(kernel),
                num_blocks,
                block_dims,
                0,
                params);
          });
        }
      } else {
        throw std::runtime_error(fmt::format(
            "Can not do binary op {} on inputs of {} with result of {}.",
            op,
            dtype_to_string(a.dtype()),
            dtype_to_string(out.dtype())));
      }
    });
  });
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  auto& encoder = cu::get_command_encoder(s);
  set_binary_op_output_data(
      a, b, out, bopt, [&](auto n) { return cu::malloc_async(n, encoder); });
  binary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define BINARY_GPU(OP)                                              \
  void OP::eval_gpu(const std::vector<array>& inputs, array& out) { \
    nvtx3::scoped_range r(#OP "::eval_gpu");                        \
    auto& s = out.primitive().stream();                             \
    binary_op_gpu<cu::OP>(inputs, out, name(), s);                  \
  }

// ============================================================================
// BINARY OPERATION IMPLEMENTATIONS
// ============================================================================

// Arithmetic operations
BINARY_GPU(Add)
BINARY_GPU(Subtract)
BINARY_GPU(Multiply)
BINARY_GPU(Divide)
BINARY_GPU(Remainder)
BINARY_GPU(Power)
BINARY_GPU(Maximum)
BINARY_GPU(Minimum)

// Comparison operations
BINARY_GPU(Greater)
BINARY_GPU(GreaterEqual)
BINARY_GPU(Less)
BINARY_GPU(LessEqual)
BINARY_GPU(NotEqual)

// Logical operations
BINARY_GPU(LogicalAnd)
BINARY_GPU(LogicalOr)

// Other binary operations
BINARY_GPU(ArcTan2)
BINARY_GPU(LogAddExp)

// Equal operation with special NaN handling
void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Equal::eval_gpu");
  auto& s = out.primitive().stream();
  if (equal_nan_) {
    binary_op_gpu<cu::NaNEqual>(inputs, out, name(), s);
  } else {
    binary_op_gpu<cu::Equal>(inputs, out, name(), s);
  }
}

// Bitwise operations
void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("BitwiseBinary::eval_gpu");
  auto& s = out.primitive().stream();
  switch (op_) {
    case BitwiseBinary::And:
      binary_op_gpu<cu::BitwiseAnd>(inputs, out, name(), s);
      break;
    case BitwiseBinary::Or:
      binary_op_gpu<cu::BitwiseOr>(inputs, out, name(), s);
      break;
    case BitwiseBinary::Xor:
      binary_op_gpu<cu::BitwiseXor>(inputs, out, name(), s);
      break;
    case BitwiseBinary::LeftShift:
      binary_op_gpu<cu::LeftShift>(inputs, out, name(), s);
      break;
    case BitwiseBinary::RightShift:
      binary_op_gpu<cu::RightShift>(inputs, out, name(), s);
      break;
  }
}

// ============================================================================
// Force instantiation of complex64 binary kernels for Windows registration
// ============================================================================

// complex64 + complex64 -> complex64 (Multiply, Add, Subtract, Divide)
// N_READS = 16 / sizeof(cu::complex64_t) = 16 / 8 = 2
template __global__ void
cu::binary_ss<cu::Multiply, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_sv<cu::Multiply, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vs<cu::Multiply, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vv<cu::Multiply, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);

template __global__ void
cu::binary_ss<cu::Add, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_sv<cu::Add, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vs<cu::Add, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vv<cu::Add, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);

template __global__ void
cu::binary_ss<cu::Subtract, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_sv<cu::Subtract, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vs<cu::Subtract, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vv<cu::Subtract, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);

template __global__ void
cu::binary_ss<cu::Divide, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_sv<cu::Divide, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vs<cu::Divide, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);
template __global__ void
cu::binary_vv<cu::Divide, cu::complex64_t, cu::complex64_t, uint32_t, 2>(
    const cu::complex64_t*,
    const cu::complex64_t*,
    cu::complex64_t*,
    uint32_t);

} // namespace mlx::core
