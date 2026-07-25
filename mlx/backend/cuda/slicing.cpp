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

  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;
  auto concurrent = encoder.concurrent_context();
  for (int i = 0; i < inputs.size(); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    auto ctype = CopyType::GeneralGeneral;
    if (axis == 0 && inputs[i].flags().row_contiguous) {
      ctype = CopyType::Vector;
    }
    copy_gpu_inplace(inputs[i], out_slice, ctype, s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const std::vector<int>& max_starts,
    const Stream& s) {
  Dtype dtype = indices.dtype();
  int nidx = axes.size();

  std::string module_name =
      fmt::format("compute_dynamic_offset_{}_{}", dtype_to_string(dtype), nidx);
  std::string kernel_name = fmt::format(
      "mlx::core::cu::compute_dynamic_offset<{}, {}>",
      dtype_to_cuda_type(dtype),
      nidx);

  auto& encoder = cu::get_command_encoder(s);

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), module_name, [&]() {
    std::string source = R"(
        #include "mlx/backend/cuda/device/utils.cuh"

        namespace mlx::core::cu {

        template <typename T, int NIDX>
        __global__ void compute_dynamic_offset(
            const T* indices,
            int64_t* offset,
            const __grid_constant__ Strides strides,
            const __grid_constant__ cuda::std::array<int, NIDX> axes,
            const __grid_constant__ cuda::std::array<int, NIDX> max_starts) {
          int64_t acc = 0;
          #pragma unroll
          for (int i = 0; i < NIDX; ++i) {
            // Clamp so the slice stays inside the operand, as XLA's
            // dynamic-slice does. Unclamped this offset addresses memory
            // outside the array entirely. The comparison is done in the
            // index's own signedness so that neither a negative signed start
            // nor a large unsigned one clamps to the wrong end of the axis.
            int64_t hi = static_cast<int64_t>(max_starts[i]);
            int64_t idx;
            if constexpr (cuda::std::numeric_limits<T>::is_signed) {
              int64_t v = static_cast<int64_t>(indices[i]);
              idx = v < 0 ? 0 : (v > hi ? hi : v);
            } else {
              uint64_t v = static_cast<uint64_t>(indices[i]);
              idx = v > static_cast<uint64_t>(hi) ? hi : static_cast<int64_t>(v);
            }
            acc += idx * strides[axes[i]];
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
    offset.set_data(cu::malloc_async(offset.itemsize(), encoder));
  }

  encoder.add_temporary(offset);
  encoder.set_input_array(indices);
  encoder.set_output_array(offset);

  cu::KernelArgs args;
  args.append(indices);
  args.append(offset);
  args.append_ndim(strides);
  args.append(axes);
  args.append(max_starts);

  auto kernel = mod.get_kernel(kernel_name);
  encoder.add_kernel_node_raw(kernel, 1, 1, {}, 0, args.args());

  return offset;
}

} // namespace mlx::core
