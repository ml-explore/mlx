// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype_utils.h"

#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

// Return pointer alignment of |x|'s data.
inline uint8_t get_alignment(const array& x) {
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(x.data<void>());
  for (; alignment < 32; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  return alignment;
}

// Convert the type of elements in |vec| to |T|.
template <typename T, typename Vec>
inline SmallVector<T> convert_vector(const Vec& vec) {
  return SmallVector<T>(vec.begin(), vec.end());
}

// Return an array that can be used as map key for |vec| with size <= MAX_NDIM.
//
// There are 2 differences from the const_param util from kernel_utils.cuh:
// 1. The rest of array is filled with 0.
// 2. This util can be used in .cpp files.
template <typename T, template <typename U> class Vec>
inline std::array<T, MAX_NDIM> vector_key(const Vec<T>& vec) {
  if (vec.size() > MAX_NDIM) {
    throw std::runtime_error(
        fmt::format("ndim can not be larger than {}.", MAX_NDIM));
  }
  std::array<T, MAX_NDIM> result = {};
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

// Helpers used by get_data_ptrs to get pointers.
inline void* get_data_ptr(const array& arr) {
  return const_cast<void*>(arr.data<void>());
}

template <typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
inline void* get_data_ptr(T& scalar) {
  return &scalar;
}

// Return an array filled with data pointers of args.
template <typename... Args>
inline std::array<void*, sizeof...(Args)> get_data_ptrs(Args&... args) {
  return {get_data_ptr(args)...};
}

// Map dtype to cudnn data type.
inline cudnnDataType_t dtype_to_cudnn_type(Dtype dtype) {
  switch (dtype) {
    case int8:
      return CUDNN_DATA_INT8;
    case int32:
      return CUDNN_DATA_INT32;
    case uint8:
      return CUDNN_DATA_UINT8;
    case float16:
      return CUDNN_DATA_HALF;
    case bfloat16:
      return CUDNN_DATA_BFLOAT16;
    case float32:
      return CUDNN_DATA_FLOAT;
    case float64:
      return CUDNN_DATA_DOUBLE;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in Convolution: {}.", dtype_to_string(dtype)));
  }
}

// Create a tensor descriptor from |x|.
cudnn_frontend::Tensor build_cudnn_tensor(int64_t id, const array& x);

// Create a tensor descriptor from |x|, and transpose from NHWC to NCHW.
cudnn_frontend::Tensor build_cudnn_tensor_nchw(int64_t id, const array& x);

// Create a tensor descriptor from |x|, make sure it is 4D, and transpose it
// from NHWC to NCHW.
cudnn_frontend::Tensor build_cudnn_tensor_4d_nchw(int64_t id, const array& x);

// Create a 4D scalar tensor descriptor, which is passed by value.
cudnn_frontend::Tensor build_cudnn_scalar_4d(int64_t id, Dtype dtype);

// Find a working plan for |op_graph|.
std::optional<cudnn_frontend::ExecutionPlan> find_cudnn_plan_from_op_graph(
    cudnnHandle_t handle,
    cudnnBackendDescriptorType_t backend_type,
    Dtype dtype,
    cudnn_frontend::OperationGraph& op_graph);

// Encode the plan to command buffer by capturing.
bool encode_cudnn_plan_with_capturing(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    int num_args,
    const int64_t* uids,
    void** data_ptrs);

#if CUDNN_VERSION >= 90500
// Encode the plan to command buffer by using native graph api of cudnn. If the
// |graph| is empty it will be populated, otherwise it will be updated.
bool encode_cudnn_plan_with_graph_api(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    CudaGraph& graph,
    int num_args,
    const int64_t* uids,
    void** data_ptrs);
#endif

// Helpers to make calls like encode_cudnn_plan(..., {'x', 'y', 'z'}, x, y, z).
template <typename... Args>
bool encode_cudnn_plan(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    std::initializer_list<int64_t> uids,
    Args&... args) {
  assert(uids.size() == sizeof...(args));
  auto data_ptrs = get_data_ptrs(args...);
  return encode_cudnn_plan_with_capturing(
      encoder, plan, uids.size(), uids.begin(), data_ptrs.data());
}

#if CUDNN_VERSION >= 90500
template <typename... Args>
bool encode_cudnn_plan(
    cu::CommandEncoder& encoder,
    cudnn_frontend::ExecutionPlan& plan,
    CudaGraph& graph,
    std::initializer_list<int64_t> uids,
    Args&... args) {
  assert(uids.size() == sizeof...(args));
  auto data_ptrs = get_data_ptrs(args...);
  return encode_cudnn_plan_with_graph_api(
      encoder, plan, graph, uids.size(), uids.begin(), data_ptrs.data());
}
#endif

} // namespace mlx::core
