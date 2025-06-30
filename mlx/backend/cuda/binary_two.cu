// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/cucomplex_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void
binary_ss(const In* a, const In* b, Out* out_a, Out* out_b, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto out = Op{}(a[0], b[0]);
    out_a[0] = out[0];
    out_b[0] = out[1];
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void
binary_sv(const In* a, const In* b, Out* out_a, Out* out_b, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto out = Op{}(a[0], b[index]);
    out_a[index] = out[0];
    out_b[index] = out[1];
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void
binary_vs(const In* a, const In* b, Out* out_a, Out* out_b, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto out = Op{}(a[index], b[0]);
    out_a[index] = out[0];
    out_b[index] = out[1];
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void
binary_vv(const In* a, const In* b, Out* out_a, Out* out_b, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto out = Op{}(a[index], b[index]);
    out_a[index] = out[0];
    out_b[index] = out[1];
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int NDIM>
__global__ void binary_g_nd(
    const In* a,
    const In* b,
    Out* out_a,
    Out* out_b,
    IdxT size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [a_idx, b_idx] = elem_to_loc_nd<NDIM>(
        index, shape.data(), a_strides.data(), b_strides.data());
    auto out = Op{}(a[a_idx], b[b_idx]);
    out_a[index] = out[0];
    out_b[index] = out[1];
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void binary_g(
    const In* a,
    const In* b,
    Out* out_a,
    Out* out_b,
    IdxT size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
    int ndim) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [a_idx, b_idx] = elem_to_loc_4d(
        index, shape.data(), a_strides.data(), b_strides.data(), ndim);
    auto out = Op{}(a[a_idx], b[b_idx]);
    out_a[index] = out[0];
    out_b[index] = out[1];
  }
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_op() {
  if (std::is_same_v<Op, DivMod>) {
    return std::is_same_v<In, Out> &&
        (std::is_integral_v<Out> || is_floating_v<Out>);
  }
  return false;
}

} // namespace cu

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    std::string_view op,
    const Stream& s) {
  assert(inputs.size() > 1);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  auto& out_a = outputs[0];
  auto& out_b = outputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_a, bopt);
  set_binary_op_output_data(a, b, out_b, bopt);

  if (out_a.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out_a);
  encoder.set_output_array(out_b);
  encoder.launch_kernel([&](cudaStream_t stream) {
    dispatch_all_types(a.dtype(), [&](auto in_type_tag) {
      dispatch_all_types(out_a.dtype(), [&](auto out_type_tag) {
        using CTYPE_IN = MLX_GET_TYPE(in_type_tag);
        using CTYPE_OUT = MLX_GET_TYPE(out_type_tag);
        if constexpr (cu::supports_binary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
          using InType = cuda_type_t<CTYPE_IN>;
          using OutType = cuda_type_t<CTYPE_OUT>;

          auto bopt = get_binary_op_type(a, b);
          if (bopt == BinaryOpType::General) {
            dispatch_bool(
                a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
                    out_a.data_size() > INT32_MAX,
                [&](auto large) {
                  using IdxT = std::conditional_t<large(), int64_t, int32_t>;
                  Shape shape;
                  std::vector<Strides> strides;
                  std::tie(shape, strides) =
                      collapse_contiguous_dims(a, b, out_a);
                  auto& a_strides = strides[0];
                  auto& b_strides = strides[1];
                  int ndim = shape.size();
                  if (ndim <= 3) {
                    dispatch_1_2_3(ndim, [&](auto dims_constant) {
                      auto kernel = cu::binary_g_nd<
                          Op,
                          InType,
                          OutType,
                          IdxT,
                          dims_constant()>;
                      auto [num_blocks, block_dims] =
                          get_launch_args(kernel, out_a, large());
                      kernel<<<num_blocks, block_dims, 0, stream>>>(
                          a.data<InType>(),
                          b.data<InType>(),
                          out_a.data<OutType>(),
                          out_b.data<OutType>(),
                          out_a.size(),
                          const_param<dims_constant()>(shape),
                          const_param<dims_constant()>(a_strides),
                          const_param<dims_constant()>(b_strides));
                    });
                  } else {
                    auto kernel = cu::binary_g<Op, InType, OutType, IdxT>;
                    auto [num_blocks, block_dims] =
                        get_launch_args(kernel, out_a, large());
                    kernel<<<num_blocks, block_dims, 0, stream>>>(
                        a.data<InType>(),
                        b.data<InType>(),
                        out_a.data<OutType>(),
                        out_b.data<OutType>(),
                        out_a.size(),
                        const_param(shape),
                        const_param(a_strides),
                        const_param(b_strides),
                        ndim);
                  }
                });
          } else {
            dispatch_bool(out_a.data_size() > INT32_MAX, [&](auto large) {
              using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
              auto kernel = cu::binary_ss<Op, InType, OutType, IdxT>;
              if (bopt == BinaryOpType::ScalarVector) {
                kernel = cu::binary_sv<Op, InType, OutType, IdxT>;
              } else if (bopt == BinaryOpType::VectorScalar) {
                kernel = cu::binary_vs<Op, InType, OutType, IdxT>;
              } else if (bopt == BinaryOpType::VectorVector) {
                kernel = cu::binary_vv<Op, InType, OutType, IdxT>;
              }
              auto [num_blocks, block_dims] = get_launch_args(
                  kernel,
                  out_a.data_size(),
                  out_a.shape(),
                  out_a.strides(),
                  large());
              kernel<<<num_blocks, block_dims, 0, stream>>>(
                  a.data<InType>(),
                  b.data<InType>(),
                  out_a.data<OutType>(),
                  out_b.data<OutType>(),
                  out_a.data_size());
            });
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not do binary op {} on inputs of {} with result of {}.",
              op,
              dtype_to_string(a.dtype()),
              dtype_to_string(out_a.dtype())));
        }
      });
    });
  });
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt);
  set_binary_op_output_data(a, b, outputs[1], bopt);
  binary_op_gpu_inplace<Op>(inputs, outputs, op, s);
}

void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("DivMod::eval_gpu");
  auto& s = outputs[0].primitive().stream();
  binary_op_gpu<cu::DivMod>(inputs, outputs, get_primitive_string(this), s);
}

} // namespace mlx::core
