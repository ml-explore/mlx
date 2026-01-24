// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/dtype_utils.h"

#include <numeric>

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
  auto concurrent = cu::get_command_encoder(s).concurrent_context();
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

  std::string module_name =
      fmt::format("compute_dynamic_offset_{}_{}", dtype_to_string(dtype), nidx);
  std::string kernel_name = fmt::format(
      "mlx::core::cu::compute_dynamic_offset<{}, {}>",
      dtype_to_cuda_type(dtype),
      nidx);

  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::string source = R"(
        #include "mlx/backend/cuda/device/utils.cuh"

        namespace mlx::core::cu {

        template <typename T, int NIDX>
        __global__ void compute_dynamic_offset(
            const T* indices,
            int64_t* offset,
            const __grid_constant__ Strides strides,
            const __grid_constant__ cuda::std::array<int, NIDX> axes) {
          int64_t acc = 0;
          #pragma unroll
          for (int i = 0; i < NIDX; ++i) {
            acc += indices[i] * strides[axes[i]];
          }
          *offset = acc;
        }

        } // namespace mlx::core::cu
    )";
    return std::make_tuple(false, std::move(source), std::vector{kernel_name});
  });

  // Prepare output.
  array offset({1}, int64, nullptr, {});
  bool donate = indices.is_donatable() &&
      (indices.data_size() * indices.itemsize()) >= offset.itemsize();
  if (donate) {
    offset.copy_shared_buffer(indices);
  } else {
    offset.set_data(allocator::malloc(offset.itemsize()));
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.add_temporary(offset);
  encoder.set_input_array(indices);
  encoder.set_output_array(offset);

  cu::KernelArgs args;
  args.append(indices);
  args.append(offset);
  args.append_ndim(strides);
  args.append(axes);

  auto kernel = mod.get_kernel(kernel_name);
  encoder.add_kernel_node(kernel, 1, 1, 0, args.args());

  return offset;
}

} // namespace mlx::core
