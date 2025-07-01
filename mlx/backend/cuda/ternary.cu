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

template <typename Op, typename T, typename IdxT>
__global__ void
ternary_v(const bool* a, const T* b, const T* c, T* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = Op{}(a[index], b[index], c[index]);
  }
}

template <typename Op, typename T, typename IdxT, int NDIM>
__global__ void ternary_g_nd(
    const bool* a,
    const T* b,
    const T* c,
    T* out,
    IdxT size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> c_strides) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [a_idx, b_idx, c_idx] = elem_to_loc_nd<NDIM>(
        index,
        shape.data(),
        a_strides.data(),
        b_strides.data(),
        c_strides.data());
    out[index] = Op{}(a[a_idx], b[b_idx], c[c_idx]);
  }
}

template <typename Op, typename T, typename IdxT>
__global__ void ternary_g(
    const bool* a,
    const T* b,
    const T* c,
    T* out,
    IdxT size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
    const __grid_constant__ Strides c_strides,
    int ndim) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [a_idx, b_idx, c_idx] = elem_to_loc_4d(
        index,
        shape.data(),
        a_strides.data(),
        b_strides.data(),
        c_strides.data(),
        ndim);
    out[index] = Op{}(a[a_idx], b[b_idx], c[c_idx]);
  }
}

} // namespace cu

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
  encoder.launch_kernel([&](cudaStream_t stream) {
    dispatch_all_types(out.dtype(), [&](auto type_tag) {
      using DType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

      auto topt = get_ternary_op_type(a, b, c);
      if (topt == TernaryOpType::General) {
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
              if (ndim <= 3) {
                dispatch_1_2_3(ndim, [&](auto dims_constant) {
                  auto kernel =
                      cu::ternary_g_nd<Op, DType, IdxT, dims_constant()>;
                  auto [num_blocks, block_dims] =
                      get_launch_args(kernel, out, large());
                  kernel<<<num_blocks, block_dims, 0, stream>>>(
                      a.data<bool>(),
                      b.data<DType>(),
                      c.data<DType>(),
                      out.data<DType>(),
                      out.size(),
                      const_param<dims_constant()>(shape),
                      const_param<dims_constant()>(a_strides),
                      const_param<dims_constant()>(b_strides),
                      const_param<dims_constant()>(c_strides));
                });
              } else {
                auto kernel = cu::ternary_g<Op, DType, IdxT>;
                auto [num_blocks, block_dims] =
                    get_launch_args(kernel, out, large());
                kernel<<<num_blocks, block_dims, 0, stream>>>(
                    a.data<bool>(),
                    b.data<DType>(),
                    c.data<DType>(),
                    out.data<DType>(),
                    out.data_size(),
                    const_param(shape),
                    const_param(a_strides),
                    const_param(b_strides),
                    const_param(c_strides),
                    ndim);
              }
            });
      } else {
        dispatch_bool(out.data_size() > INT32_MAX, [&](auto large) {
          using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
          auto kernel = cu::ternary_v<Op, DType, IdxT>;
          auto [num_blocks, block_dims] = get_launch_args(
              kernel, out.data_size(), out.shape(), out.strides(), large());
          kernel<<<num_blocks, block_dims, 0, stream>>>(
              a.data<bool>(),
              b.data<DType>(),
              c.data<DType>(),
              out.data<DType>(),
              out.data_size());
        });
      }
    });
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
  set_ternary_op_output_data(a, b, c, out, topt);
  ternary_op_gpu_inplace<Op>(inputs, out, s);
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("select::eval_gpu");
  auto& s = out.primitive().stream();
  ternary_op_gpu<cu::Select>(inputs, out, s);
}

} // namespace mlx::core
