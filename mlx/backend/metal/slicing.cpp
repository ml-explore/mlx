// Copyright © 2024 Apple Inc.

#include <numeric>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"

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
  auto& d = metal::device(s.device);
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto concurrent_ctx = compute_encoder.start_concurrent();
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
  auto& d = metal::device(s.device);

  // Kernel to compute offset here.
  array offset({1}, int64, nullptr, {});
  bool donate = indices.is_donatable() &&
      (indices.data_size() * indices.itemsize()) >= offset.itemsize();
  if (donate) {
    offset.copy_shared_buffer(indices);
  } else {
    offset.set_data(allocator::malloc(offset.itemsize()));
  }
  d.add_temporary(offset, s.index);

  auto dtype = indices.dtype();
  std::string lib_name = "compute_dynamic_offset_" + type_to_name(dtype);
  auto lib = d.get_library(lib_name, [dtype]() {
    return fmt::format(
        R"(
        [[kernel]] void compute_dynamic_offset_{0}(
            constant const {1}* indices [[buffer(0)]],
            device int64_t& offset [[buffer(1)]],
            constant const int64_t* strides [[buffer(2)]],
            constant const int* axes [[buffer(3)]],
            constant const int& n_axes [[buffer(4)]],
            uint index [[thread_position_in_grid]]) {{
          int64_t acc = 0;
          for (int i = 0; i < n_axes; ++i) {{
            acc += indices[i] * strides[axes[i]];
          }}
          offset = acc;
        }})",
        type_to_name(dtype),
        get_type_string(dtype));
  });
  auto kernel = d.get_kernel(lib_name, lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(indices, 0);
  compute_encoder.set_output_array(offset, 1);
  compute_encoder.set_vector_bytes(strides, 2);
  compute_encoder.set_vector_bytes(axes, 3);
  int n_axes = axes.size();
  compute_encoder.set_bytes(n_axes, 4);
  MTL::Size dims = MTL::Size(1, 1, 1);
  compute_encoder.dispatch_threads(dims, dims);
  return offset;
}

} // namespace mlx::core
