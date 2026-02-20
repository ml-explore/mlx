// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/scan.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include "cuda_jit_sources.h"

#include <cuda.h>
#include <fmt/format.h>
#include <nvrtc.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>
#include <numeric>

namespace mlx::core {

namespace {

constexpr const char* g_scatter_ops[] = {"Max", "Min", "Sum", "Prod", "Assign"};

void append_indices_arg(
    cu::KernelArgs& args,
    const std::vector<array>& inputs,
    int nidx,
    int idx_ndim) {
  SmallVector<const void*> indices(nidx);
  for (int i = 0; i < nidx; ++i) {
    indices[i] = gpu_ptr<void>(inputs[i + 1]);
  }
  args.append(std::move(indices));
  SmallVector<int32_t> indices_shape(nidx * idx_ndim);
  for (int i = 0; i < nidx; ++i) {
    std::copy_n(
        inputs[i + 1].shape().begin(),
        idx_ndim,
        indices_shape.data() + i * idx_ndim);
  }
  args.append(std::move(indices_shape));
  SmallVector<int64_t> indices_strides(nidx * idx_ndim);
  for (int i = 0; i < nidx; ++i) {
    std::copy_n(
        inputs[i + 1].strides().begin(),
        idx_ndim,
        indices_strides.data() + i * idx_ndim);
  }
  args.append(std::move(indices_strides));
}

} // namespace

namespace cu {

template <typename T, bool src_contiguous, typename IdxT>
__global__ void masked_assign(
    const bool* mask,
    const int32_t* scatter_offsets,
    const T* src,
    T* out,
    IdxT total,
    const __grid_constant__ Shape src_shape,
    const __grid_constant__ Strides src_strides,
    int32_t src_ndim,
    IdxT src_batch_size,
    IdxT mask_batch_size) {
  IdxT block_id = static_cast<IdxT>(blockIdx.x) +
      static_cast<IdxT>(gridDim.x) *
          (static_cast<IdxT>(blockIdx.y) +
           static_cast<IdxT>(gridDim.y) * static_cast<IdxT>(blockIdx.z));
  IdxT thread_id = block_id * blockDim.x + threadIdx.x;
  IdxT stride =
      static_cast<IdxT>(blockDim.x) * gridDim.x * gridDim.y * gridDim.z;

  for (IdxT idx = thread_id; idx < total; idx += stride) {
    if (!mask[idx]) {
      continue;
    }

    IdxT src_index = static_cast<IdxT>(scatter_offsets[idx]);
    if (src_index >= src_batch_size) {
      // Match Metal backend behavior by skipping out-of-range source reads.
      continue;
    }

    IdxT batch_idx = idx / mask_batch_size;
    if constexpr (src_contiguous) {
      out[idx] = src[batch_idx * src_batch_size + src_index];
    } else {
      IdxT src_elem = batch_idx * src_batch_size + src_index;
      IdxT src_loc =
          elem_to_loc(src_elem, src_shape.data(), src_strides.data(), src_ndim);
      out[idx] = src[src_loc];
    }
  }
}

} // namespace cu

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Gather::eval_gpu");
  assert(inputs.size() > 0);
  const auto& src = inputs[0];

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }

  int nidx = inputs.size() - 1;
  Dtype idx_dtype = nidx > 0 ? inputs[1].dtype() : int32;
  int32_t idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  bool large = (nidx > 0 && inputs[1].size() > INT32_MAX) ||
      (src.size() > INT32_MAX) || (out.size() > INT32_MAX);

  uint32_t slice_size = std::accumulate(
      slice_sizes_.begin(), slice_sizes_.end(), 1, std::multiplies<uint32_t>());

  std::string module_name = fmt::format(
      "gather_{}_{}_{}",
      dtype_to_string(out.dtype()),
      dtype_to_string(idx_dtype),
      nidx);

  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int ndim = 0; ndim <= MAX_NDIM; ++ndim) {
      for (int large = 0; large <= 1; ++large) {
        kernel_names.push_back(
            fmt::format(
                "mlx::core::cu::gather<{}, {}, {}, {}, {}>",
                dtype_to_cuda_type(out.dtype()),
                dtype_to_cuda_type(idx_dtype),
                nidx,
                ndim,
                large ? "int64_t" : "int32_t"));
      }
    }
    return std::make_tuple(false, jit_source_gather, std::move(kernel_names));
  });

  cu::KernelArgs args;
  args.append(src);
  args.append(out);
  if (large) {
    args.append<int64_t>(out.size());
  } else {
    args.append<int32_t>(out.size());
  }
  args.append_ndim(src.shape());
  args.append_ndim(src.strides());
  args.append<int32_t>(src.ndim());
  args.append_ndim(slice_sizes_);
  args.append(slice_size);
  args.append(axes_);
  append_indices_arg(args, inputs, nidx, idx_ndim);

  std::string kernel_name = fmt::format(
      "mlx::core::cu::gather<{}, {}, {}, {}, {}>",
      dtype_to_cuda_type(out.dtype()),
      dtype_to_cuda_type(idx_dtype),
      nidx,
      idx_ndim,
      large ? "int64_t" : "int32_t");

  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);

  auto kernel = mod.get_kernel(kernel_name);
  auto [num_blocks, block_dims] = get_launch_args(out, large);
  encoder.add_kernel_node_raw(
      kernel, num_blocks, block_dims, {}, 0, args.args());
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Gather::eval_gpu");
  assert(inputs.size() > 1);
  auto& upd = inputs.back();

  // Copy src into out.
  CopyType copy_type;
  if (inputs[0].data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (inputs[0].flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(inputs[0], out, copy_type);

  // Empty update.
  if (upd.size() == 0) {
    return;
  }

  int nidx = axes_.size();
  Dtype idx_dtype = nidx > 0 ? inputs[1].dtype() : int32;
  int32_t idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  bool large = (nidx > 0 && inputs[1].size() > INT32_MAX) ||
      (upd.size() > INT32_MAX) || (out.size() > INT32_MAX);

  int32_t upd_post_idx_size = std::accumulate(
      upd.shape().begin() + idx_ndim,
      upd.shape().end(),
      1,
      std::multiplies<int32_t>());

  const char* op = g_scatter_ops[reduce_type_];
  std::string module_name = fmt::format(
      "scatter_{}_{}_{}_{}",
      dtype_to_string(out.dtype()),
      dtype_to_string(idx_dtype),
      op,
      nidx);

  auto& s = stream();
  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int ndim = 0; ndim <= MAX_NDIM; ++ndim) {
      for (int large = 0; large <= 1; ++large) {
        kernel_names.push_back(
            fmt::format(
                "mlx::core::cu::scatter<{}, {}, mlx::core::cu::Scatter{}, {}, {}, {}>",
                dtype_to_cuda_type(out.dtype()),
                dtype_to_cuda_type(idx_dtype),
                op,
                nidx,
                ndim,
                large ? "int64_t" : "int32_t"));
      }
    }
    return std::make_tuple(false, jit_source_scatter, std::move(kernel_names));
  });

  cu::KernelArgs args;
  args.append(upd);
  args.append(out);
  if (large) {
    args.append<int64_t>(upd.size());
  } else {
    args.append<int32_t>(upd.size());
  }
  args.append_ndim(upd.shape());
  args.append_ndim(upd.strides());
  args.append<int32_t>(upd.ndim());
  if (large) {
    args.append<int64_t>(upd_post_idx_size);
  } else {
    args.append<int32_t>(upd_post_idx_size);
  }
  args.append_ndim(out.shape());
  args.append_ndim(out.strides());
  args.append<int32_t>(out.ndim());
  args.append(axes_);
  append_indices_arg(args, inputs, nidx, idx_ndim);

  std::string kernel_name = fmt::format(
      "mlx::core::cu::scatter<{}, {}, mlx::core::cu::Scatter{}, {}, {}, {}>",
      dtype_to_cuda_type(out.dtype()),
      dtype_to_cuda_type(idx_dtype),
      op,
      nidx,
      idx_ndim,
      large ? "int64_t" : "int32_t");

  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  auto kernel = mod.get_kernel(kernel_name);
  auto [num_blocks, block_dims] = get_launch_args(upd, large);
  encoder.add_kernel_node_raw(
      kernel, num_blocks, block_dims, {}, 0, args.args());
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("GatherAxis::eval_gpu");
  assert(inputs.size() > 1);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }

  bool large = idx.size() > INT32_MAX || src.size() > INT32_MAX;

  std::string module_name = fmt::format(
      "gather_axis_{}_{}",
      dtype_to_string(out.dtype()),
      dtype_to_string(idx.dtype()));

  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int ndim = 0; ndim <= MAX_NDIM; ++ndim) {
      for (int contiguous = 0; contiguous < 4; ++contiguous) {
        for (int large = 0; large <= 1; ++large) {
          kernel_names.push_back(
              fmt::format(
                  "mlx::core::cu::gather_axis<{}, {}, {}, {}, {}, {}>",
                  dtype_to_cuda_type(out.dtype()),
                  dtype_to_cuda_type(idx.dtype()),
                  ndim,
                  contiguous & 1 ? true : false,
                  contiguous & 2 ? true : false,
                  large ? "int64_t" : "int32_t"));
        }
      }
    }
    return std::make_tuple(
        false, jit_source_gather_axis, std::move(kernel_names));
  });

  size_t idx_size_pre = 1;
  size_t idx_size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    idx_size_pre *= idx.shape(i);
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    idx_size_post *= idx.shape(i);
  }
  size_t idx_size_axis = idx.shape(axis_);

  cu::KernelArgs args;
  args.append(src);
  args.append(idx);
  args.append(out);
  if (large) {
    args.append<int64_t>(idx_size_pre);
    args.append<int64_t>(idx_size_axis);
    args.append<int64_t>(idx_size_post);
  } else {
    args.append<int32_t>(idx_size_pre);
    args.append<int32_t>(idx_size_axis);
    args.append<int32_t>(idx_size_post);
  }
  args.append(remove_index(idx.shape(), axis_));
  args.append(remove_index(src.strides(), axis_));
  args.append(remove_index(idx.strides(), axis_));
  args.append<int32_t>(axis_);
  args.append(src.shape(axis_));
  args.append(src.strides(axis_));
  args.append(idx.strides(axis_));

  std::string kernel_name = fmt::format(
      "mlx::core::cu::gather_axis<{}, {}, {}, {}, {}, {}>",
      dtype_to_cuda_type(out.dtype()),
      dtype_to_cuda_type(idx.dtype()),
      src.ndim() - 1,
      src.flags().row_contiguous,
      idx.flags().row_contiguous,
      large ? "int64_t" : "int32_t");

  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  auto kernel = mod.get_kernel(kernel_name);
  auto [num_blocks, block_dims] = get_launch_args(idx, large);
  encoder.add_kernel_node_raw(
      kernel, num_blocks, block_dims, {}, 0, args.args());
}

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("ScatterAxis::eval_gpu");
  assert(inputs.size() > 2);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];
  const auto& upd = inputs[2];

  // Copy src into out.
  CopyType copy_type;
  if (src.data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (src.flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(src, out, copy_type);

  // Empty update.
  if (upd.size() == 0) {
    return;
  }

  bool large = idx.size() > INT32_MAX || src.size() > INT32_MAX;

  const char* op = reduce_type_ == ScatterAxis::Sum ? "Sum" : "Assign";
  std::string module_name = fmt::format(
      "scatter_axis_{}_{}_{}",
      dtype_to_string(out.dtype()),
      dtype_to_string(idx.dtype()),
      op);

  auto& s = stream();
  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int ndim = 0; ndim <= MAX_NDIM; ++ndim) {
      for (int contiguous = 0; contiguous < 4; ++contiguous) {
        for (int large = 0; large <= 1; ++large) {
          kernel_names.push_back(
              fmt::format(
                  "mlx::core::cu::scatter_axis<{}, {}, mlx::core::cu::Scatter{}, {}, {}, {}, {}>",
                  dtype_to_cuda_type(out.dtype()),
                  dtype_to_cuda_type(idx.dtype()),
                  op,
                  ndim,
                  contiguous & 1 ? true : false,
                  contiguous & 2 ? true : false,
                  large ? "int64_t" : "int32_t"));
        }
      }
    }
    return std::make_tuple(
        false, jit_source_scatter_axis, std::move(kernel_names));
  });

  size_t idx_size_pre = 1;
  size_t idx_size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    idx_size_pre *= idx.shape(i);
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    idx_size_post *= idx.shape(i);
  }
  size_t idx_size_axis = idx.shape(axis_);

  cu::KernelArgs args;
  args.append(upd);
  args.append(idx);
  args.append(out);
  if (large) {
    args.append<int64_t>(idx_size_pre);
    args.append<int64_t>(idx_size_axis);
    args.append<int64_t>(idx_size_post);
  } else {
    args.append<int32_t>(idx_size_pre);
    args.append<int32_t>(idx_size_axis);
    args.append<int32_t>(idx_size_post);
  }
  args.append(remove_index(idx.shape(), axis_));
  args.append(remove_index(upd.strides(), axis_));
  args.append(remove_index(idx.strides(), axis_));
  args.append<int32_t>(axis_);
  args.append(out.shape(axis_));
  args.append(upd.strides(axis_));
  args.append(idx.strides(axis_));

  std::string kernel_name = fmt::format(
      "mlx::core::cu::scatter_axis<{}, {}, mlx::core::cu::Scatter{}, {}, {}, {}, {}>",
      dtype_to_cuda_type(out.dtype()),
      dtype_to_cuda_type(idx.dtype()),
      op,
      idx.ndim() - 1,
      upd.flags().row_contiguous,
      idx.flags().row_contiguous,
      large ? "int64_t" : "int32_t");

  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  auto kernel = mod.get_kernel(kernel_name);
  auto [num_blocks, block_dims] = get_launch_args(idx, large);
  encoder.add_kernel_node_raw(
      kernel, num_blocks, block_dims, {}, 0, args.args());
}

void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("MaskedScatter::eval_gpu");
  assert(inputs.size() == 3);

  const array& dst = inputs[0];
  const array& mask = inputs[1];
  const array& src = inputs[2];

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  const size_t total = mask.size();
  const CopyType copy_type = (total == 1)
      ? CopyType::Scalar
      : (dst.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  copy_gpu(dst, out, copy_type, s);
  if (total == 0) {
    return;
  }

  array mask_flat = flatten_in_eval(mask, 1, -1, s);
  if (mask_flat.data<void>() != mask.data<void>()) {
    encoder.add_temporary(mask_flat);
  }
  if (!mask_flat.flags().row_contiguous) {
    mask_flat = contiguous_copy_gpu(mask_flat, s);
    encoder.add_temporary(mask_flat);
  }

  array scatter_offsets(mask_flat.shape(), int32, nullptr, {});
  scatter_offsets.set_data(cu::malloc_async(scatter_offsets.nbytes(), encoder));
  encoder.add_temporary(scatter_offsets);

  scan_gpu_inplace(
      mask_flat,
      scatter_offsets,
      Scan::Sum,
      /* axis= */ 1,
      /* reverse= */ false,
      /* inclusive= */ false,
      s);

  const size_t batch_count = mask.shape(0);
  const size_t mask_batch_size = mask_flat.size() / batch_count;
  const size_t src_batch_size = src.size() / src.shape(0);

  encoder.set_input_array(mask_flat);
  encoder.set_input_array(scatter_offsets);
  encoder.set_input_array(src);
  encoder.set_output_array(out);

  dispatch_all_types(out.dtype(), [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_bool(src.flags().row_contiguous, [&](auto src_contiguous) {
      dispatch_bool(
          total > INT32_MAX || src.size() > INT32_MAX, [&](auto large) {
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;
            auto [num_blocks, block_dims] = get_launch_args(
                mask_flat.size(),
                mask_flat.shape(),
                mask_flat.strides(),
                large());
            auto kernel = cu::masked_assign<T, src_contiguous.value, IdxT>;
            encoder.add_kernel_node(
                kernel,
                num_blocks,
                block_dims,
                0,
                gpu_ptr<bool>(mask_flat),
                gpu_ptr<int32_t>(scatter_offsets),
                gpu_ptr<T>(src),
                gpu_ptr<T>(out),
                static_cast<IdxT>(mask_flat.size()),
                const_param(src.shape()),
                const_param(src.strides()),
                static_cast<int32_t>(src.ndim()),
                static_cast<IdxT>(src_batch_size),
                static_cast<IdxT>(mask_batch_size));
          });
    });
  });
}

} // namespace mlx::core
