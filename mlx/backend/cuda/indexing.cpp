// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/scan.h"
#include "mlx/backend/gpu/slicing.h"
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
constexpr const char* g_slice_ops[] =
    {"Maximum", "Minimum", "Add", "Multiply", ""};

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
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
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
  bool large = total > INT32_MAX || src.size() > INT32_MAX;
  bool vectorized = src.flags().row_contiguous && dst.flags().row_contiguous;
  constexpr int kMaskedScatterVecSize = 16;
  constexpr int kMaskedScatterVecBlockDim = 256;

  std::string module_name =
      fmt::format("masked_scatter_{}", dtype_to_string(out.dtype()));
  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int src_contiguous = 0; src_contiguous <= 1; ++src_contiguous) {
      for (int dst_contiguous = 0; dst_contiguous <= 1; ++dst_contiguous) {
        for (int use_large = 0; use_large <= 1; ++use_large) {
          kernel_names.push_back(
              fmt::format(
                  "mlx::core::cu::masked_scatter<{}, {}, {}, {}>",
                  dtype_to_cuda_type(out.dtype()),
                  src_contiguous ? "true" : "false",
                  dst_contiguous ? "true" : "false",
                  use_large ? "int64_t" : "int32_t"));
        }
      }
    }
    for (int use_large = 0; use_large <= 1; ++use_large) {
      kernel_names.push_back(
          fmt::format(
              "mlx::core::cu::masked_scatter_vec_contiguous<{}, {}, {}>",
              dtype_to_cuda_type(out.dtype()),
              use_large ? "int64_t" : "int32_t",
              kMaskedScatterVecSize));
    }
    return std::make_tuple(false, jit_source_scatter, std::move(kernel_names));
  });

  cu::KernelArgs args;
  args.append(dst);
  args.append(mask_flat);
  args.append(scatter_offsets);
  args.append(src);
  args.append(out);
  if (large) {
    args.append<int64_t>(mask_flat.size());
    args.append<int64_t>(src_batch_size);
    args.append<int64_t>(mask_batch_size);
  } else {
    args.append<int32_t>(mask_flat.size());
    args.append<int32_t>(src_batch_size);
    args.append<int32_t>(mask_batch_size);
  }
  if (!vectorized) {
    args.append_ndim(dst.shape());
    args.append_ndim(dst.strides());
    args.append<int32_t>(dst.ndim());
    args.append_ndim(src.shape());
    args.append_ndim(src.strides());
    args.append<int32_t>(src.ndim());
  }

  encoder.set_input_array(dst);
  encoder.set_input_array(mask_flat);
  encoder.set_input_array(scatter_offsets);
  encoder.set_input_array(src);
  encoder.set_output_array(out);

  std::string kernel_name = vectorized
      ? fmt::format(
            "mlx::core::cu::masked_scatter_vec_contiguous<{}, {}, {}>",
            dtype_to_cuda_type(out.dtype()),
            large ? "int64_t" : "int32_t",
            kMaskedScatterVecSize)
      : fmt::format(
            "mlx::core::cu::masked_scatter<{}, {}, {}, {}>",
            dtype_to_cuda_type(out.dtype()),
            src.flags().row_contiguous ? "true" : "false",
            dst.flags().row_contiguous ? "true" : "false",
            large ? "int64_t" : "int32_t");
  auto kernel = mod.get_kernel(kernel_name);
  auto [num_blocks, block_dims] = vectorized
      ? get_launch_args(
            mask_flat, large, kMaskedScatterVecSize, kMaskedScatterVecBlockDim)
      : get_launch_args(mask_flat, large);
  encoder.add_kernel_node_raw(
      kernel, num_blocks, block_dims, {}, 0, args.args());
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("SliceUpdate::eval_gpu");
  assert(inputs.size() == 2);
  if (out.size() == 0) {
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());

  // Calculate out strides, initial offset and if copy needs to be made
  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);

  // Do copy for None reduce type
  if (reduce_type_ == SliceUpdate::None) {
    copy_gpu_inplace(
        /* const array& src = */ upd,
        /* array& dst = */ out,
        /* const Shape& data_shape = */ upd.shape(),
        /* const Strides& i_strides = */ upd.strides(),
        /* const Strides& o_strides = */ out_strides,
        /* int64_t i_offset = */ 0,
        /* int64_t o_offset = */ data_offset,
        /* CopyType ctype = */ CopyType::GeneralGeneral,
        /* const Stream& s = */ stream());
    return;
  }

  auto [shape, strides] =
      collapse_contiguous_dims(upd.shape(), {upd.strides(), out_strides});
  int nwork = 1;
  if (shape.back() % 4 == 0) {
    nwork = 4;
  } else if (shape.back() % 2 == 0) {
    nwork = 2;
  }

  const char* op_name = g_slice_ops[reduce_type_];
  auto [ds, rc, cc] = check_contiguity(shape, strides[1]);
  bool upd_contiguous = upd.flags().row_contiguous;
  bool upd_scalar = upd.data_size() == 1;
  bool out_contiguous = rc;
  bool large = upd.size() > INT32_MAX;
  std::string module_name =
      fmt::format("slice_update_{}_{}", op_name, dtype_to_string(out.dtype()));

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int out_c = 0; out_c <= 1; ++out_c) {
      for (int upd_c = 0; upd_c <= 1; ++upd_c) {
        for (int upd_s = 0; upd_s <= 1; ++upd_s) {
          for (int large = 0; large <= 1; ++large) {
            for (int nwork = 1; nwork <= 16; nwork *= 2) {
              kernel_names.push_back(
                  fmt::format(
                      "mlx::core::cu::slice_update_op<{}, {}, mlx::core::cu::{}, {}, {}, {}, {}>",
                      dtype_to_cuda_type(out.dtype()),
                      large ? "int64_t" : "int32_t",
                      op_name,
                      out_c ? "true" : "false",
                      upd_c ? "true" : "false",
                      upd_s ? "true" : "false",
                      nwork));
            }
          }
        }
      }
    }
    return std::make_tuple(
        false, jit_source_slice_update, std::move(kernel_names));
  });

  cu::KernelArgs args;
  args.append(upd);
  args.append(out);
  args.append<int64_t>(upd.size());
  args.append_ndim(shape);
  args.append_ndim(strides[0]);
  args.append<int32_t>(shape.size());
  args.append_ndim(strides[1]);
  args.append<int64_t>(data_offset);

  encoder.set_input_array(upd);
  encoder.set_output_array(out);

  std::string kernel_name;
  kernel_name = fmt::format(
      "mlx::core::cu::slice_update_op<{}, {}, mlx::core::cu::{}, {}, {}, {}, {}>",
      dtype_to_cuda_type(out.dtype()),
      large ? "int64_t" : "int32_t",
      op_name,
      out_contiguous,
      upd_contiguous,
      upd_scalar,
      nwork);

  auto kernel = mod.get_kernel(kernel_name);
  auto [num_blocks, block_dims] = get_launch_args(upd, large, nwork);
  encoder.add_kernel_node_raw(
      kernel, num_blocks, block_dims, {}, 0, args.args());
}

} // namespace mlx::core
