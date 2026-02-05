// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/rocm/device.h"
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
  module_name_ss << "compute_dynamic_offset_" << dtype_to_string(dtype) << "_" << nidx;
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

        namespace mlx::core::rocm {

        template <typename T, int NIDX>
        __global__ void compute_dynamic_offset(
            const T* indices,
            int64_t* offset,
            const int64_t* strides,
            const int* axes) {
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
    offset.set_data(allocator::malloc(offset.itemsize()));
  }

  encoder.add_temporary(offset);
  encoder.set_input_array(indices);
  encoder.set_output_array(offset);

  // Copy strides and axes to device
  array strides_arr({static_cast<int>(strides.size())}, int64);
  array axes_arr({static_cast<int>(axes.size())}, int32);
  strides_arr.set_data(allocator::malloc(strides_arr.nbytes()));
  axes_arr.set_data(allocator::malloc(axes_arr.nbytes()));
  encoder.add_temporary(strides_arr);
  encoder.add_temporary(axes_arr);

  // Get kernel before launching to avoid any potential issues
  auto kernel = mod.get_kernel(kernel_name);

  // Get GPU pointers before lambda to avoid synchronization issues
  const void* indices_ptr = gpu_ptr<void>(indices);
  void* offset_ptr = gpu_ptr<void>(offset);
  void* strides_arr_ptr = gpu_ptr<void>(strides_arr);
  void* axes_arr_ptr = gpu_ptr<void>(axes_arr);

  encoder.launch_kernel([&, kernel, indices_ptr, offset_ptr, strides_arr_ptr, axes_arr_ptr](hipStream_t stream) {
    fprintf(stderr, "DEBUG: Starting hipMemcpyAsync for strides\n");
    (void)hipMemcpyAsync(
        strides_arr_ptr,
        strides.data(),
        strides.size() * sizeof(int64_t),
        hipMemcpyHostToDevice,
        stream);
    fprintf(stderr, "DEBUG: Starting hipMemcpyAsync for axes\n");
    (void)hipMemcpyAsync(
        axes_arr_ptr,
        axes.data(),
        axes.size() * sizeof(int32_t),
        hipMemcpyHostToDevice,
        stream);

    fprintf(stderr, "DEBUG: Launching kernel\n");
    void* args[] = {
        const_cast<void*>(indices_ptr),
        offset_ptr,
        strides_arr_ptr,
        axes_arr_ptr
    };
    (void)hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr);
    fprintf(stderr, "DEBUG: Kernel launched\n");
  });
  
  fprintf(stderr, "DEBUG: compute_dynamic_offset returning\n");

  return offset;
}

} // namespace mlx::core
