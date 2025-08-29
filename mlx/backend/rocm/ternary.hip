// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/ternary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/ternary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <rocthrust/device_ptr.h>
#include <rocthrust/transform.h>

namespace mlx::core {

namespace rocm {

template <typename Op, typename Condition, typename A, typename B, typename Out>
constexpr bool supports_ternary_op() {
  if (std::is_same_v<Op, Select>) {
    return std::is_same_v<Condition, bool> && std::is_same_v<A, Out> && std::is_same_v<B, Out>;
  }
  return false;
}

} // namespace rocm

template <typename Op>
void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op,
    const Stream& s) {
  auto& condition = inputs[0];
  auto& a = inputs[1];
  auto& b = inputs[2];
  
  if (condition.size() == 0) {
    return;
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(condition);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    MLX_SWITCH_ALL_TYPES(condition.dtype(), CONDITION_TYPE, {
      MLX_SWITCH_ALL_TYPES(a.dtype(), A_TYPE, {
        MLX_SWITCH_ALL_TYPES(b.dtype(), B_TYPE, {
          MLX_SWITCH_ALL_TYPES(out.dtype(), OUT_TYPE, {
            if constexpr (rocm::supports_ternary_op<Op, CONDITION_TYPE, A_TYPE, B_TYPE, OUT_TYPE>()) {
              using ConditionType = hip_type_t<CONDITION_TYPE>;
              using AType = hip_type_t<A_TYPE>;
              using BType = hip_type_t<B_TYPE>;
              using OutType = hip_type_t<OUT_TYPE>;
              
              auto policy = rocm::thrust_policy(stream);
              auto condition_ptr = rocthrust::device_pointer_cast(condition.data<ConditionType>());
              auto a_ptr = rocthrust::device_pointer_cast(a.data<AType>());
              auto b_ptr = rocthrust::device_pointer_cast(b.data<BType>());
              auto out_ptr = rocthrust::device_pointer_cast(out.data<OutType>());
              
              if (condition.flags().contiguous && a.flags().contiguous && b.flags().contiguous) {
                auto ternary_op = [=] __device__ (const auto& tuple) -> OutType {
                  return Op{}(rocthrust::get<0>(tuple), rocthrust::get<1>(tuple), rocthrust::get<2>(tuple));
                };
                
                auto zip_begin = rocthrust::make_zip_iterator(
                    rocthrust::make_tuple(condition_ptr, a_ptr, b_ptr));
                auto zip_end = rocthrust::make_zip_iterator(
                    rocthrust::make_tuple(condition_ptr + condition.data_size(), 
                                         a_ptr + a.data_size(), 
                                         b_ptr + b.data_size()));
                
                rocthrust::transform(policy, zip_begin, zip_end, out_ptr, ternary_op);
              } else {
                // Handle non-contiguous arrays with general iterators
                auto [condition_shape, condition_strides] = collapse_contiguous_dims(condition);
                auto [a_shape, a_strides] = collapse_contiguous_dims(a);
                auto [b_shape, b_strides] = collapse_contiguous_dims(b);
                
                auto [condition_begin, condition_end] = rocm::make_general_iterators<int64_t>(
                    condition_ptr, condition.size(), condition_shape, condition_strides);
                auto [a_begin, a_end] = rocm::make_general_iterators<int64_t>(
                    a_ptr, a.size(), a_shape, a_strides);
                auto [b_begin, b_end] = rocm::make_general_iterators<int64_t>(
                    b_ptr, b.size(), b_shape, b_strides);
                
                auto ternary_op = [=] __device__ (const auto& tuple) -> OutType {
                  return Op{}(rocthrust::get<0>(tuple), rocthrust::get<1>(tuple), rocthrust::get<2>(tuple));
                };
                
                auto zip_begin = rocthrust::make_zip_iterator(
                    rocthrust::make_tuple(condition_begin, a_begin, b_begin));
                auto zip_end = rocthrust::make_zip_iterator(
                    rocthrust::make_tuple(condition_end, a_end, b_end));
                
                rocthrust::transform(policy, zip_begin, zip_end, out_ptr, ternary_op);
              }
            } else {
              throw std::runtime_error(fmt::format(
                  "Can not do ternary op {} on inputs of {}, {}, {} with output of {}.",
                  op,
                  dtype_to_string(condition.dtype()),
                  dtype_to_string(a.dtype()),
                  dtype_to_string(b.dtype()),
                  dtype_to_string(out.dtype())));
            }
          });
        });
      });
    });
  });
}

template <typename Op>
void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op,
    const Stream& s) {
  set_ternary_output_data(inputs, out);
  ternary_op_gpu_inplace<Op>(inputs, out, op, s);
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  ternary_op_gpu<rocm::Select>(inputs, out, get_primitive_string(this), s);
}

} // namespace mlx::core

__global__ void select_kernel(float* condition, float* a, float* b, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = (condition[idx] != 0.0f) ? a[idx] : b[idx];
  }
}

void launch_select(float* condition, float* a, float* b, float* output, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(select_kernel, dim3(blocks), dim3(threads), 0, stream, condition, a, b, output, n);
}

} // namespace mlx::core::rocm 