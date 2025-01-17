// Copyright © 2023-2024 Apple Inc.
#include <format>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/jit/indexing.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

constexpr int METAL_MAX_INDEX_ARRAYS = 20;

std::pair<std::string, std::string> make_index_args(
    const std::string& idx_type,
    int nidx) {
  std::ostringstream idx_args;
  std::ostringstream idx_arr;
  for (int i = 0; i < nidx; ++i) {
    idx_args << std::format(
        "const device {0} *idx{1} [[buffer({2})]],", idx_type, i, 20 + i);
    idx_arr << std::format("idx{0}", i);
    if (i < nidx - 1) {
      idx_args << "\n";
      idx_arr << ",";
    }
  }
  return {idx_args.str(), idx_arr.str()};
}

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& src = inputs[0];
  int nidx = inputs.size() - 1;

  if (nidx > METAL_MAX_INDEX_ARRAYS) {
    std::ostringstream msg;
    msg << "[Gather::eval_gpu] Gathering with more than "
        << METAL_MAX_INDEX_ARRAYS << " index arrays not yet supported.";
    throw std::runtime_error(msg.str());
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  size_t ndim = src.ndim();

  bool large_index = nidx && inputs[1].size() > INT32_MAX;
  bool large_src = src.size() > INT32_MAX;
  bool large_out = out.size() > INT32_MAX;
  bool large = large_index || large_src || large_out;

  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";
  std::string kernel_name = std::format(
      "gather{0}{1}_{2}_{3}_{4}",
      type_to_name(out),
      idx_type_name,
      nidx,
      idx_ndim,
      large ? "int64_t" : "int");
  std::string lib_name = kernel_name;

  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source = metal::utils();
    kernel_source += metal::gather();
    std::string out_type_str = get_type_string(out.dtype());
    std::string idx_type_str =
        nidx ? get_type_string(inputs[1].dtype()) : "bool";
    auto [idx_args, idx_arr] = make_index_args(idx_type_str, nidx);

    // Index dimension specializations
    kernel_source += std::format(
        gather_kernels,
        type_to_name(out) + idx_type_name,
        out_type_str,
        idx_type_str,
        nidx,
        idx_args,
        idx_arr,
        idx_ndim,
        large ? "int64_t" : "int");
    return kernel_source;
  });

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name, lib);
  compute_encoder.set_compute_pipeline_state(kernel);

  size_t slice_size = 1;
  for (auto s : slice_sizes_) {
    slice_size *= s;
  }

  // Launch 3D grid of threads
  // First two dimensions for the indices, the last one for the slice
  size_t dim0 = 1;
  size_t dim1 = 1;
  if (nidx) {
    if (inputs[1].ndim() >= 1) {
      dim0 = inputs[1].shape(0);
    }
    if (inputs[1].ndim() >= 2) {
      dim1 = inputs[1].size() / dim0;
    }
  }
  size_t dim2 = slice_size;
  auto group_dims = get_block_dims(dim0, dim1, dim2);
  MTL::Size grid_dims = MTL::Size(dim0, dim1, dim2);

  // Collect all idx shapes and strides into one place
  std::vector<int> idx_shapes;
  std::vector<size_t> idx_strides;
  std::vector<char> idx_contigs;
  for (int i = 0; i < nidx; ++i) {
    idx_shapes.insert(
        idx_shapes.end(),
        inputs[i + 1].shape().begin(),
        inputs[i + 1].shape().end());
    idx_strides.insert(
        idx_strides.end(),
        inputs[i + 1].strides().begin(),
        inputs[i + 1].strides().end());
    idx_contigs.push_back(inputs[i + 1].flags().row_contiguous);
  }

  // Set all the buffers
  compute_encoder.set_input_array(src, 0);
  compute_encoder.set_output_array(out, 1);

  // Set source info
  compute_encoder.set_vector_bytes(src.shape(), 2);
  compute_encoder.set_vector_bytes(src.strides(), 3);
  compute_encoder.set_bytes(ndim, 4);
  compute_encoder.set_vector_bytes(slice_sizes_, 5);
  compute_encoder.set_vector_bytes(axes_, 6);

  // Set index info
  //
  // We don't need to check for empty idx_shapes because gather has a
  // idx_ndim == 0 specialization
  compute_encoder.set_vector_bytes(idx_shapes, 7);
  compute_encoder.set_vector_bytes(idx_strides, 8);
  compute_encoder.set_vector_bytes(idx_contigs, 9);
  compute_encoder.set_bytes(idx_ndim, 10);

  // Set index buffers
  for (int i = 0; i < nidx; ++i) {
    compute_encoder.set_input_array(inputs[i + 1], 20 + i);
  }

  // Launch grid
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (size_of(out.dtype()) == 8) {
    std::ostringstream msg;
    msg << "[Scatter::eval_gpu] Does not support " << out.dtype();
    throw std::invalid_argument(msg.str());
  }

  int nidx = axes_.size();
  if (nidx > METAL_MAX_INDEX_ARRAYS) {
    std::ostringstream msg;
    msg << "[Scatter::eval_gpu] Gathering with more than "
        << METAL_MAX_INDEX_ARRAYS << " index arrays not yet supported.";
    throw std::runtime_error(msg.str());
  }

  // Copy src into out
  CopyType copy_type;
  if (inputs[0].data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (inputs[0].flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(inputs[0], out, copy_type);

  auto& upd = inputs.back();

  // Empty update
  if (upd.size() == 0) {
    return;
  }

  // Get stream
  auto& s = stream();
  auto& d = metal::device(s.device);

  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  size_t idx_size = nidx ? inputs[1].size() : 1;

  auto idx_to_out = idx_size / out.size();
  int nwork;
  if (idx_ndim <= 1 || idx_to_out < 1) {
    nwork = 1;
  } else if (idx_to_out <= 4) {
    nwork = 4;
  } else if (idx_to_out < 16) {
    nwork = 8;
  } else if (idx_to_out < 32) {
    nwork = 16;
  } else {
    nwork = 32;
  }

  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";
  std::string op_name;
  switch (reduce_type_) {
    case Scatter::None:
      op_name = "none";
      break;
    case Scatter::Sum:
      op_name = "sum";
      break;
    case Scatter::Prod:
      op_name = "prod";
      break;
    case Scatter::Max:
      op_name = "max";
      break;
    case Scatter::Min:
      op_name = "min";
      break;
  }
  auto upd_contig = upd.flags().row_contiguous;
  bool large_out = out.size() > INT32_MAX;
  bool large_idx = nidx && (inputs[1].size() > INT32_MAX);
  bool large_upd = upd.size() > INT32_MAX;
  bool large = large_out || large_idx || large_upd;
  std::string kernel_name = std::format(
      "scatter{0}{1}_{2}_{3}_{4}_nwork{5}_{6}",
      type_to_name(out),
      idx_type_name,
      op_name,
      nidx,
      upd_contig ? "updc_true" : "updc_false",
      nwork,
      large ? "int64_t" : "int");
  std::string lib_name = kernel_name;

  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source = metal::utils();
    concatenate(kernel_source, metal::reduce_utils(), metal::scatter());

    std::string out_type_str = get_type_string(out.dtype());
    std::string idx_type_str =
        nidx ? get_type_string(inputs[1].dtype()) : "bool";
    std::string op_type;
    switch (reduce_type_) {
      case Scatter::None:
        op_type = "None";
        break;
      case Scatter::Sum:
        op_type = "Sum<{0}>";
        break;
      case Scatter::Prod:
        op_type = "Prod<{0}>";
        break;
      case Scatter::Max:
        op_type = "Max<{0}>";
        break;
      case Scatter::Min:
        op_type = "Min<{0}>";
        break;
    }
    if (reduce_type_ != Scatter::None) {
      op_type = std::vformat(op_type, std::make_format_args(out_type_str));
    }
    auto [idx_args, idx_arr] = make_index_args(idx_type_str, nidx);

    kernel_source += std::format(
        scatter_kernels,
        type_to_name(out) + idx_type_name + "_" + op_name,
        out_type_str,
        idx_type_str,
        op_type,
        nidx,
        idx_args,
        idx_arr,
        upd_contig,
        nwork,
        large ? "int64_t" : "int");
    return kernel_source;
  });

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name, lib);

  size_t nthreads = upd.size();

  compute_encoder.set_compute_pipeline_state(kernel);

  // Set all the buffers
  compute_encoder.set_input_array(upd, 1);
  compute_encoder.set_output_array(out, 2);

  // Set update info
  size_t upd_ndim = upd.ndim();
  size_t upd_size = 1;
  for (int i = idx_ndim; i < upd.ndim(); ++i) {
    upd_size *= upd.shape(i);
  }
  // Collect all idx shapes and strides into one place
  Shape idx_shapes;
  Strides idx_strides;
  // To access .data() use char instead of bool
  // bool is 1 byte in Metal so this is safe
  std::vector<char> idx_contigs;
  for (int i = 0; i < nidx; ++i) {
    idx_shapes.insert(
        idx_shapes.end(),
        inputs[i + 1].shape().begin(),
        inputs[i + 1].shape().end());
    idx_strides.insert(
        idx_strides.end(),
        inputs[i + 1].strides().begin(),
        inputs[i + 1].strides().end());
    idx_contigs.push_back(inputs[i + 1].flags().row_contiguous);
  }

  if (upd_ndim == 0) {
    // Need placeholders so Metal doesn't compalain
    int shape_ = 0;
    int64_t stride_ = 0;
    compute_encoder.set_bytes(shape_, 3);
    compute_encoder.set_bytes(stride_, 4);
  } else {
    compute_encoder.set_vector_bytes(upd.shape(), 3);
    compute_encoder.set_vector_bytes(upd.strides(), 4);
  }
  compute_encoder.set_bytes(upd_ndim, 5);
  compute_encoder.set_bytes(upd_size, 6);

  // Set output info
  size_t out_ndim = out.ndim();
  if (out_ndim == 0) {
    // Need placeholders so Metal doesn't compalain
    int shape_ = 0;
    int64_t stride_ = 0;
    compute_encoder.set_bytes(shape_, 7);
    compute_encoder.set_bytes(stride_, 8);
  } else {
    compute_encoder.set_vector_bytes(out.shape(), 7);
    compute_encoder.set_vector_bytes(out.strides(), 8);
  }
  compute_encoder.set_bytes(out_ndim, 9);
  compute_encoder.set_vector_bytes(axes_, 10);

  // Set index info
  if (idx_ndim == 0) {
    // Add a 0 in idx_shapes and strides to avoid the missing buffer binding
    // error in the metal API.
    idx_shapes.push_back(0);
    idx_strides.push_back(0);
    idx_contigs.push_back(false);
  }
  compute_encoder.set_vector_bytes(idx_shapes, 11);
  compute_encoder.set_vector_bytes(idx_strides, 12);
  compute_encoder.set_vector_bytes(idx_contigs, 13);
  compute_encoder.set_bytes(idx_ndim, 14);
  compute_encoder.set_bytes(idx_size, 15);

  // Set index buffers
  for (int i = 0; i < nidx; ++i) {
    compute_encoder.set_input_array(inputs[i + 1], 20 + i);
  }

  // Launch grid
  auto grid_y = (nthreads / upd_size);
  grid_y = (grid_y + nwork - 1) / nwork;
  MTL::Size grid_dims = MTL::Size(upd_size, grid_y, 1);
  auto thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size != 1024) {
    throw std::runtime_error("[Scatter::eval_gpu] Invalid number of threads");
  }
  MTL::Size group_dims = get_block_dims(upd_size, grid_y, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core
