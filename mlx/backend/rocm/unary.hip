// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/unary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace mlx::core {

namespace rocm {

namespace cg = cooperative_groups;

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void unary_v(const In* in, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = Op{}(in[i + j]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = Op{}(in[j]);
      }
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void unary_g(
    const In* in,
    Out* out,
    IdxT size_rest,
    const int* shape,
    const int64_t* strides,
    int ndim) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto stride_x = strides[ndim - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Compute base offset for this row using elem_to_loc style calculation
  // elem = index_rest * shape_x gives us the linear element index for the start of this row
  IdxT elem = index_rest * shape_x;
  IdxT idx = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    idx += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  
  // Process elements in this row
  for (IdxT i = index_x * N_READS; i < shape_x; i += blockDim.x * gridDim.x * N_READS) {
    if (i + N_READS <= shape_x) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        IdxT in_idx = idx + (i + j) * stride_x;
        out[shape_x * index_rest + i + j] = Op{}(in[in_idx]);
      }
    } else {
      for (IdxT j = i; j < shape_x; ++j) {
        IdxT in_idx = idx + j * stride_x;
        out[shape_x * index_rest + j] = Op{}(in[in_idx]);
      }
    }
  }
}

template <typename Op, typename In, typename Out>
constexpr bool supports_unary_op() {
  if constexpr (std::is_same_v<Op, Abs> || std::is_same_v<Op, Negative> ||
      std::is_same_v<Op, Sign> || std::is_same_v<Op, Square>) {
    return std::is_same_v<In, Out>;
  }
  if constexpr (std::is_same_v<Op, ArcCosh> || std::is_same_v<Op, ArcSinh> ||
      std::is_same_v<Op, ArcTanh> || std::is_same_v<Op, Erf> ||
      std::is_same_v<Op, ErfInv> || std::is_same_v<Op, Expm1> ||
      std::is_same_v<Op, Sigmoid>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if constexpr (std::is_same_v<Op, BitwiseInvert>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  if constexpr (std::is_same_v<Op, Ceil> || std::is_same_v<Op, Floor>) {
    return std::is_same_v<In, Out> && !is_complex_v<In>;
  }
  if constexpr (std::is_same_v<Op, Conjugate>) {
    return std::is_same_v<In, Out> && is_complex_v<In>;
  }
  if constexpr (std::is_same_v<Op, ArcCos> || std::is_same_v<Op, ArcSin> ||
      std::is_same_v<Op, ArcTan> || std::is_same_v<Op, Cos> ||
      std::is_same_v<Op, Cosh> || std::is_same_v<Op, Exp> ||
      std::is_same_v<Op, Log> || std::is_same_v<Op, Log2> ||
      std::is_same_v<Op, Log10> || std::is_same_v<Op, Log1p> ||
      std::is_same_v<Op, Round> || std::is_same_v<Op, Rsqrt> ||
      std::is_same_v<Op, Sqrt> || std::is_same_v<Op, Sin> ||
      std::is_same_v<Op, Sinh> || std::is_same_v<Op, Tan> ||
      std::is_same_v<Op, Tanh>) {
    return std::is_same_v<In, Out> && is_inexact_v<In>;
  }
  if constexpr (std::is_same_v<Op, Imag> || std::is_same_v<Op, Real>) {
    return is_complex_v<In> && std::is_same_v<Out, float>;
  }
  if constexpr (std::is_same_v<Op, LogicalNot>) {
    return std::is_same_v<In, Out> && std::is_same_v<In, bool>;
  }
  return false;
}

} // namespace rocm

template <typename Op>
void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& in = inputs[0];
  if (in.size() == 0) {
    return;
  }
  bool contig = in.flags().contiguous;
  bool large;
  if (!contig) {
    large = in.data_size() > INT32_MAX || out.size() > INT32_MAX;
  } else {
    large = in.data_size() > UINT32_MAX;
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Dispatch based on input and output types
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using CTYPE_IN = MLX_GET_TYPE(in_type_tag);
      using CTYPE_OUT = MLX_GET_TYPE(out_type_tag);
      using InType = hip_type_t<CTYPE_IN>;
      using OutType = hip_type_t<CTYPE_OUT>;
      
      if constexpr (rocm::supports_unary_op<Op, InType, OutType>()) {
        if (contig) {
          // Contiguous case - use unary_v
          constexpr int N_READS = 4;
          int block_size = 256;
          auto size = out.data_size();
          int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
          num_blocks = std::min(num_blocks, 65535);
          
          encoder.launch_kernel([&](hipStream_t stream) {
            if (large) {
              hipLaunchKernelGGL(
                  (rocm::unary_v<Op, InType, OutType, int64_t, N_READS>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  gpu_ptr<const InType>(in), gpu_ptr<OutType>(out), static_cast<int64_t>(size));
            } else {
              hipLaunchKernelGGL(
                  (rocm::unary_v<Op, InType, OutType, uint32_t, N_READS>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  gpu_ptr<const InType>(in), gpu_ptr<OutType>(out), static_cast<uint32_t>(size));
            }
          });
        } else {
          // Non-contiguous case - use unary_g with strided access
          auto [shape_vec, strides_vec] = collapse_contiguous_dims(in);
          int ndim = shape_vec.size();
          
          // Allocate device memory for shape and strides
          array shape_arr({ndim}, int32, nullptr, {});
          array strides_arr({ndim}, int64, nullptr, {});
          shape_arr.set_data(allocator::malloc(shape_arr.nbytes()));
          strides_arr.set_data(allocator::malloc(strides_arr.nbytes()));
          encoder.add_temporary(shape_arr);
          encoder.add_temporary(strides_arr);
          
          // Copy shape and strides to vectors for capture
          std::vector<int32_t> shape_copy(shape_vec.begin(), shape_vec.end());
          std::vector<int64_t> strides_copy(strides_vec.begin(), strides_vec.end());
          
          int dim0 = ndim > 0 ? shape_vec.back() : 1;
          size_t rest = out.size() / dim0;
          
          constexpr int N_READS = 4;
          int work_per_thread = (dim0 >= 4) ? 4 : 1;
          dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
          
          // Calculate block and grid dimensions
          int block_x = std::min(dim0, 32);
          int block_y = std::min(static_cast<int>(rest), 256 / block_x);
          int num_blocks_x = (dim0 + block_x - 1) / block_x;
          int num_blocks_y = (rest + block_y - 1) / block_y;
          
          encoder.launch_kernel([=, &in, &out, &shape_arr, &strides_arr](hipStream_t stream) {
            // Copy shape and strides to device
            (void)hipMemcpyAsync(
                shape_arr.data<int32_t>(),
                shape_copy.data(),
                ndim * sizeof(int32_t),
                hipMemcpyHostToDevice,
                stream);
            (void)hipMemcpyAsync(
                strides_arr.data<int64_t>(),
                strides_copy.data(),
                ndim * sizeof(int64_t),
                hipMemcpyHostToDevice,
                stream);
            
            if (large) {
              if (work_per_thread == 4) {
                hipLaunchKernelGGL(
                    (rocm::unary_g<Op, InType, OutType, int64_t, 4>),
                    dim3(num_blocks_x, num_blocks_y), dim3(block_x, block_y), 0, stream,
                    gpu_ptr<const InType>(in), gpu_ptr<OutType>(out),
                    static_cast<int64_t>(rest),
                    shape_arr.data<int32_t>(),
                    strides_arr.data<int64_t>(),
                    ndim);
              } else {
                hipLaunchKernelGGL(
                    (rocm::unary_g<Op, InType, OutType, int64_t, 1>),
                    dim3(num_blocks_x, num_blocks_y), dim3(block_x, block_y), 0, stream,
                    gpu_ptr<const InType>(in), gpu_ptr<OutType>(out),
                    static_cast<int64_t>(rest),
                    shape_arr.data<int32_t>(),
                    strides_arr.data<int64_t>(),
                    ndim);
              }
            } else {
              if (work_per_thread == 4) {
                hipLaunchKernelGGL(
                    (rocm::unary_g<Op, InType, OutType, int32_t, 4>),
                    dim3(num_blocks_x, num_blocks_y), dim3(block_x, block_y), 0, stream,
                    gpu_ptr<const InType>(in), gpu_ptr<OutType>(out),
                    static_cast<int32_t>(rest),
                    shape_arr.data<int32_t>(),
                    strides_arr.data<int64_t>(),
                    ndim);
              } else {
                hipLaunchKernelGGL(
                    (rocm::unary_g<Op, InType, OutType, int32_t, 1>),
                    dim3(num_blocks_x, num_blocks_y), dim3(block_x, block_y), 0, stream,
                    gpu_ptr<const InType>(in), gpu_ptr<OutType>(out),
                    static_cast<int32_t>(rest),
                    shape_arr.data<int32_t>(),
                    strides_arr.data<int64_t>(),
                    ndim);
              }
            }
          });
        }
      }
    });
  });
}

template <typename Op>
void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  set_unary_output_data(inputs[0], out);
  unary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define UNARY_GPU(func)                                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    auto& s = out.primitive().stream();                               \
    unary_op_gpu<rocm::func>(inputs, out, name(), s);                 \
  }

UNARY_GPU(Abs)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcTan)
UNARY_GPU(ArcTanh)
UNARY_GPU(BitwiseInvert)
UNARY_GPU(Ceil)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Floor)
UNARY_GPU(Imag)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Negative)
UNARY_GPU(Real)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Square)
UNARY_GPU(Tan)
UNARY_GPU(Tanh)

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  switch (base_) {
    case Base::e:
      unary_op_gpu<rocm::Log>(inputs, out, name(), s);
      break;
    case Base::two:
      unary_op_gpu<rocm::Log2>(inputs, out, name(), s);
      break;
    case Base::ten:
      unary_op_gpu<rocm::Log10>(inputs, out, name(), s);
      break;
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  auto& s = out.primitive().stream();
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu<rocm::Round>(inputs, out, name(), s);
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<rocm::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<rocm::Sqrt>(inputs, out, "Sqrt", s);
  }
}

} // namespace mlx::core
