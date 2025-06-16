// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include "cuda_jit_sources.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>
#include <numeric>

namespace mlx::core {

namespace {

constexpr const char* g_scatter_ops[] = {"Max", "Min", "Sum", "Prod", "Assign"};

void append_indices_arg(
    cu::JitModule& mod,
    const std::vector<array>& inputs,
    int nidx,
    int idx_ndim) {
  std::vector<const void*> indices(nidx);
  for (int i = 0; i < nidx; ++i) {
    indices[i] = inputs[i + 1].data<void>();
  }
  mod.append_arg(std::move(indices));
  std::vector<int32_t> indices_shape(nidx * idx_ndim);
  for (int i = 0; i < nidx; ++i) {
    std::copy_n(
        inputs[i + 1].shape().begin(),
        idx_ndim,
        indices_shape.data() + i * idx_ndim);
  }
  mod.append_arg(std::move(indices_shape));
  std::vector<int64_t> indices_strides(nidx * idx_ndim);
  for (int i = 0; i < nidx; ++i) {
    std::copy_n(
        inputs[i + 1].strides().begin(),
        idx_ndim,
        indices_strides.data() + i * idx_ndim);
  }
  mod.append_arg(std::move(indices_strides));
}

} // namespace

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Gather::eval_gpu");
  assert(inputs.size() > 0);
  const auto& src = inputs[0];

  out.set_data(allocator::malloc(out.nbytes()));
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

  auto& s = stream();
  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int ndim = 0; ndim <= MAX_NDIM; ++ndim) {
      for (int large = 0; large <= 1; ++large) {
        kernel_names.push_back(fmt::format(
            "mlx::core::cu::gather<{}, {}, {}, {}, {}>",
            dtype_to_cuda_type(out.dtype()),
            dtype_to_cuda_type(idx_dtype),
            nidx,
            ndim,
            large ? "int64_t" : "int32_t"));
      }
    }
    return std::make_pair(jit_source_gather, std::move(kernel_names));
  });

  mod.append_arg(src);
  mod.append_arg(out);
  if (large) {
    mod.append_arg<int64_t>(out.size());
  } else {
    mod.append_arg<int32_t>(out.size());
  }
  mod.append_ndim_arg(src.shape());
  mod.append_ndim_arg(src.strides());
  mod.append_arg<int32_t>(src.ndim());
  mod.append_ndim_arg(slice_sizes_);
  mod.append_arg(slice_size);
  mod.append_arg(axes_);
  append_indices_arg(mod, inputs, nidx, idx_ndim);

  std::string kernel_name = fmt::format(
      "mlx::core::cu::gather<{}, {}, {}, {}, {}>",
      dtype_to_cuda_type(out.dtype()),
      dtype_to_cuda_type(idx_dtype),
      nidx,
      idx_ndim,
      large ? "int64_t" : "int32_t");

  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    mod.launch_kernel(stream, kernel_name, out, large);
  });
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
        kernel_names.push_back(fmt::format(
            "mlx::core::cu::scatter<{}, {}, mlx::core::cu::Scatter{}, {}, {}, {}>",
            dtype_to_cuda_type(out.dtype()),
            dtype_to_cuda_type(idx_dtype),
            op,
            nidx,
            ndim,
            large ? "int64_t" : "int32_t"));
      }
    }
    return std::make_pair(jit_source_scatter, std::move(kernel_names));
  });

  mod.append_arg(upd);
  mod.append_arg(out);
  if (large) {
    mod.append_arg<int64_t>(upd.size());
  } else {
    mod.append_arg<int32_t>(upd.size());
  }
  mod.append_ndim_arg(upd.shape());
  mod.append_ndim_arg(upd.strides());
  mod.append_arg<int32_t>(upd.ndim());
  if (large) {
    mod.append_arg<int64_t>(upd_post_idx_size);
  } else {
    mod.append_arg<int32_t>(upd_post_idx_size);
  }
  mod.append_ndim_arg(out.shape());
  mod.append_ndim_arg(out.strides());
  mod.append_arg<int32_t>(out.ndim());
  mod.append_arg(axes_);
  append_indices_arg(mod, inputs, nidx, idx_ndim);

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
  encoder.launch_kernel([&](cudaStream_t stream) {
    mod.launch_kernel(stream, kernel_name, upd, large);
  });
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("GatherAxis::eval_gpu");
  assert(inputs.size() > 1);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  bool large = idx.size() > INT32_MAX || src.size() > INT32_MAX;

  std::string module_name = fmt::format(
      "gather_axis_{}_{}",
      dtype_to_string(out.dtype()),
      dtype_to_string(idx.dtype()));

  auto& s = stream();
  cu::JitModule& mod = cu::get_jit_module(s.device, module_name, [&]() {
    std::vector<std::string> kernel_names;
    for (int ndim = 0; ndim <= MAX_NDIM; ++ndim) {
      for (int contiguous = 0; contiguous < 4; ++contiguous) {
        for (int large = 0; large <= 1; ++large) {
          kernel_names.push_back(fmt::format(
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
    return std::make_pair(jit_source_gather_axis, std::move(kernel_names));
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

  mod.append_arg(src);
  mod.append_arg(idx);
  mod.append_arg(out);
  if (large) {
    mod.append_arg<int64_t>(idx_size_pre);
    mod.append_arg<int64_t>(idx_size_axis);
    mod.append_arg<int64_t>(idx_size_post);
  } else {
    mod.append_arg<int32_t>(idx_size_pre);
    mod.append_arg<int32_t>(idx_size_axis);
    mod.append_arg<int32_t>(idx_size_post);
  }
  mod.append_arg(remove_index(idx.shape(), axis_));
  mod.append_arg(remove_index(src.strides(), axis_));
  mod.append_arg(remove_index(idx.strides(), axis_));
  mod.append_arg<int32_t>(axis_);
  mod.append_arg(src.shape(axis_));
  mod.append_arg(src.strides(axis_));
  mod.append_arg(idx.strides(axis_));

  std::string kernel_name = fmt::format(
      "mlx::core::cu::gather_axis<{}, {}, {}, {}, {}, {}>",
      dtype_to_cuda_type(out.dtype()),
      dtype_to_cuda_type(idx.dtype()),
      src.ndim() - 1,
      src.flags().row_contiguous,
      idx.flags().row_contiguous,
      large ? "int64_t" : "int32_t");

  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    mod.launch_kernel(stream, kernel_name, idx, large);
  });
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
          kernel_names.push_back(fmt::format(
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
    return std::make_pair(jit_source_scatter_axis, std::move(kernel_names));
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

  mod.append_arg(upd);
  mod.append_arg(idx);
  mod.append_arg(out);
  if (large) {
    mod.append_arg<int64_t>(idx_size_pre);
    mod.append_arg<int64_t>(idx_size_axis);
    mod.append_arg<int64_t>(idx_size_post);
  } else {
    mod.append_arg<int32_t>(idx_size_pre);
    mod.append_arg<int32_t>(idx_size_axis);
    mod.append_arg<int32_t>(idx_size_post);
  }
  mod.append_arg(remove_index(idx.shape(), axis_));
  mod.append_arg(remove_index(upd.strides(), axis_));
  mod.append_arg(remove_index(idx.strides(), axis_));
  mod.append_arg<int32_t>(axis_);
  mod.append_arg(out.shape(axis_));
  mod.append_arg(upd.strides(axis_));
  mod.append_arg(idx.strides(axis_));

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
  encoder.launch_kernel([&](cudaStream_t stream) {
    mod.launch_kernel(stream, kernel_name, idx, large);
  });
}

} // namespace mlx::core
