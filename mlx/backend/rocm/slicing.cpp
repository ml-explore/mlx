// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/utils.h"
#include "mlx/dtype_utils.h"

#include <numeric>
#include <sstream>

namespace mlx::core {

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;
  for (int i = 0; i < inputs.size(); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  Dtype dtype = indices.dtype();
  int nidx = axes.size();

  std::ostringstream module_name_ss;
  module_name_ss << "compute_dynamic_offset_" << dtype_to_string(dtype) << "_"
                 << nidx;
  std::string module_name = module_name_ss.str();

  std::ostringstream kernel_name_ss;
  kernel_name_ss << "mlx::core::rocm::compute_dynamic_offset<"
                 << dtype_to_hip_type(dtype) << ", " << nidx << ">";
  std::string kernel_name = kernel_name_ss.str();

  rocm::JitModule& mod = rocm::get_jit_module(s.device, module_name, [&]() {
    std::ostringstream source;
    source << R"(
        #include <hip/hip_runtime.h>

        // Standard type definitions for JIT compilation
        using int64_t = signed long long;
        using int32_t = signed int;

        #define MAX_NDIM 10

        namespace mlx::core::rocm {

        template <typename T, int N>
        struct hip_array {
          T data_[N];
          __host__ __device__ T& operator[](int i) { return data_[i]; }
          __host__ __device__ const T& operator[](int i) const {
            return data_[i];
          }
        };

        template <typename T, int NIDX>
        __global__ void compute_dynamic_offset(
            const T* indices,
            int64_t* offset,
            hip_array<int64_t, MAX_NDIM> strides,
            hip_array<int32_t, MAX_NDIM> axes) {
          int64_t acc = 0;
          #pragma unroll
          for (int i = 0; i < NIDX; ++i) {
            acc += static_cast<int64_t>(indices[i]) * strides[axes[i]];
          }
          *offset = acc;
        }

        } // namespace mlx::core::rocm
    )";
    return std::make_tuple(false, source.str(), std::vector{kernel_name});
  });

  auto& encoder = rocm::get_command_encoder(s);
  // Prepare output.
  array offset({1}, int64, nullptr, {});
  bool donate = indices.is_donatable() &&
      (indices.data_size() * indices.itemsize()) >= offset.itemsize();
  if (donate) {
    offset.copy_shared_buffer(indices);
  } else {
    offset.set_data(mlx::core::rocm::malloc_async(offset.itemsize(), encoder));
  }

  encoder.add_temporary(offset);
  encoder.set_input_array(indices);
  encoder.set_output_array(offset);

  rocm::hip_array<int64_t, MAX_NDIM> strides_arg = {};
  rocm::hip_array<int32_t, MAX_NDIM> axes_arg = {};
  for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
    strides_arg.data_[i] = static_cast<int64_t>(strides[i]);
  }
  for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
    axes_arg.data_[i] = static_cast<int32_t>(axes[i]);
  }

  // Get kernel before launching to avoid any potential issues
  auto kernel = mod.get_kernel(kernel_name);

  // Get GPU pointers before lambda to avoid synchronization issues
  const void* indices_ptr = gpu_ptr<void>(indices);
  void* offset_ptr = gpu_ptr<void>(offset);

  encoder.launch_kernel(
      [kernel, indices_ptr, offset_ptr, strides_arg, axes_arg](
          hipStream_t stream) {
        const void* arg0 = indices_ptr;
        void* arg1 = offset_ptr;
        rocm::hip_array<int64_t, MAX_NDIM> arg2 = strides_arg;
        rocm::hip_array<int32_t, MAX_NDIM> arg3 = axes_arg;
        void* args[] = {&arg0, &arg1, &arg2, &arg3};
        (void)hipModuleLaunchKernel(
            kernel, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr);
      });

  return offset;
}

} // namespace mlx::core
