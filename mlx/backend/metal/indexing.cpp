// Copyright © 2023-2024 Apple Inc.
#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/compiled_preamble.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

constexpr int METAL_MAX_INDEX_ARRAYS = 20;

constexpr std::string_view gather_preamble = R"(
template <typename T, typename IdxT, int NIDX, int IDX_NDIM>
METAL_FUNC void gather_impl(
    const device T* src [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant int* src_shape [[buffer(2)]],
    const constant size_t* src_strides [[buffer(3)]],
    const constant size_t& src_ndim [[buffer(4)]],
    const constant int* slice_sizes [[buffer(5)]],
    const constant int* axes [[buffer(6)]],
    const thread Indices<IdxT, NIDX>& indices,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto ind_idx = index.x;
  auto ind_offset = index.y;

  size_t src_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    size_t idx_loc;
    if (IDX_NDIM == 0) {
      idx_loc = 0;
    } else if (IDX_NDIM == 1) {
      idx_loc = ind_idx * indices.strides[indices.ndim * i];
    } else {
      idx_loc = elem_to_loc(
          ind_idx,
          &indices.shapes[indices.ndim * i],
          &indices.strides[indices.ndim * i],
          indices.ndim);
    }
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], src_shape[ax]);
    src_idx += idx_val * src_strides[ax];
  }

  auto src_offset = elem_to_loc(ind_offset, slice_sizes, src_strides, src_ndim);

  size_t out_idx = index.y + static_cast<size_t>(grid_dim.y) * index.x;
  out[out_idx] = src[src_offset + src_idx];
}
)";

constexpr std::string_view gather_kernels = R"(
[[kernel]] void gather{0}_{3}_{6}(
    const device {1}* src [[buffer(0)]],
    device {1}* out [[buffer(1)]],
    const constant int* src_shape [[buffer(2)]],
    const constant size_t* src_strides [[buffer(3)]],
    const constant size_t& src_ndim [[buffer(4)]],
    const constant int* slice_sizes [[buffer(5)]],
    const constant int* axes [[buffer(6)]],
    const constant int* idx_shapes [[buffer(7)]],
    const constant size_t* idx_strides [[buffer(8)]],
    const constant int& idx_ndim [[buffer(9)]],
    {4}
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {{
  Indices<{2}, {3}> idxs{{
    {{ {5} }}, idx_shapes, idx_strides, idx_ndim}};

  return gather_impl<{1}, {2}, {3}, {6}>(
      src,
      out,
      src_shape,
      src_strides,
      src_ndim,
      slice_sizes,
      axes,
      idxs,
      index,
      grid_dim);
}}
)";

constexpr std::string_view scatter_preamble = R"(
template <typename T, typename IdxT, typename Op, int NIDX>
METAL_FUNC void scatter_1d_index_impl(
    const device T* updates [[buffer(1)]],
    device mlx_atomic<T>* out [[buffer(2)]],
    const constant int* out_shape [[buffer(3)]],
    const constant size_t* out_strides [[buffer(4)]],
    const constant size_t& upd_size [[buffer(5)]],
    const thread array<const device IdxT*, NIDX>& idx_buffers,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;

  uint out_idx = 0;
  for (int i = 0; i < NIDX; i++) {
    auto idx_val = offset_neg_idx(idx_buffers[i][gid.y], out_shape[i]);
    out_idx += idx_val * out_strides[i];
  }

  op.atomic_update(out, updates[gid.y * upd_size + gid.x], out_idx + gid.x);
}

template <typename T, typename IdxT, typename Op, int NIDX>
METAL_FUNC void scatter_impl(
    const device T* updates [[buffer(1)]],
    device mlx_atomic<T>* out [[buffer(2)]],
    const constant int* upd_shape [[buffer(3)]],
    const constant size_t* upd_strides [[buffer(4)]],
    const constant size_t& upd_ndim [[buffer(5)]],
    const constant size_t& upd_size [[buffer(6)]],
    const constant int* out_shape [[buffer(7)]],
    const constant size_t* out_strides [[buffer(8)]],
    const constant size_t& out_ndim [[buffer(9)]],
    const constant int* axes [[buffer(10)]],
    const thread Indices<IdxT, NIDX>& indices,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;
  auto ind_idx = gid.y;
  auto ind_offset = gid.x;

  size_t out_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    auto idx_loc = elem_to_loc(
        ind_idx,
        &indices.shapes[indices.ndim * i],
        &indices.strides[indices.ndim * i],
        indices.ndim);
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], out_shape[ax]);
    out_idx += idx_val * out_strides[ax];
  }

  if (upd_size > 1) {
    auto out_offset = elem_to_loc(
        ind_offset, upd_shape + indices.ndim, out_strides, out_ndim);
    out_idx += out_offset;
  }

  auto upd_idx =
      elem_to_loc(gid.y * upd_size + gid.x, upd_shape, upd_strides, upd_ndim);
  op.atomic_update(out, updates[upd_idx], out_idx);
}
)";

constexpr std::string_view scatter_kernels = R"(
[[kernel]] void scatter_1d_index{0}_{4}(
    const device {1}* updates [[buffer(1)]],
    device mlx_atomic<{1}>* out [[buffer(2)]],
    const constant int* out_shape [[buffer(3)]],
    const constant size_t* out_strides [[buffer(4)]],
    const constant size_t& upd_size [[buffer(5)]],
    {5}
    uint2 gid [[thread_position_in_grid]]) {{
  const array<const device {2}*, {4}> idx_buffers = {{ {6} }};
  return scatter_1d_index_impl<{1}, {2}, {3}, {4}>(
      updates, out, out_shape, out_strides, upd_size, idx_buffers, gid);
}}

[[kernel]] void scatter{0}_{4}(
    const device {1}* updates [[buffer(1)]],
    device mlx_atomic<{1}>* out [[buffer(2)]],
    const constant int* upd_shape [[buffer(3)]],
    const constant size_t* upd_strides [[buffer(4)]],
    const constant size_t& upd_ndim [[buffer(5)]],
    const constant size_t& upd_size [[buffer(6)]],
    const constant int* out_shape [[buffer(7)]],
    const constant size_t* out_strides [[buffer(8)]],
    const constant size_t& out_ndim [[buffer(9)]],
    const constant int* axes [[buffer(10)]],
    const constant int* idx_shapes [[buffer(11)]],
    const constant size_t* idx_strides [[buffer(12)]],
    const constant int& idx_ndim [[buffer(13)]],
    {5}
    uint2 gid [[thread_position_in_grid]]) {{
  Indices<{2}, {4}> idxs{{ {{ {6} }}, idx_shapes, idx_strides, idx_ndim}};

  return scatter_impl<{1}, {2}, {3}, {4}>(
      updates,
      out,
      upd_shape,
      upd_strides,
      upd_ndim,
      upd_size,
      out_shape,
      out_strides,
      out_ndim,
      axes,
      idxs,
      gid);
}}
)";

std::pair<std::string, std::string> make_index_args(
    const std::string& idx_type,
    int nidx) {
  std::ostringstream idx_args;
  std::ostringstream idx_arr;
  for (int i = 0; i < nidx; ++i) {
    idx_args << fmt::format(
        "const device {0} *idx{1} [[buffer({2})]],", idx_type, i, 20 + i);
    idx_arr << fmt::format("idx{0}", i);
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

  std::string lib_name;
  std::string kernel_name;
  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";
  {
    std::ostringstream kname;
    kname << "gather" << type_to_name(out) << idx_type_name << "_" << nidx
          << "_" << idx_ndim;
    lib_name = kname.str();
    kernel_name = lib_name;
  }

  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::get_kernel_preamble();
    kernel_source << gather_preamble;

    std::string out_type_str = get_type_string(out.dtype());
    std::string idx_type_str =
        nidx ? get_type_string(inputs[1].dtype()) : "bool";
    auto [idx_args, idx_arr] = make_index_args(idx_type_str, nidx);

    // Index dimension specializations
    kernel_source << fmt::format(
        gather_kernels,
        type_to_name(out) + idx_type_name,
        out_type_str,
        idx_type_str,
        nidx,
        idx_args,
        idx_arr,
        idx_ndim);
    lib = d.get_library(lib_name, kernel_source.str());
  }

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name, lib);
  compute_encoder->setComputePipelineState(kernel);

  size_t slice_size = 1;
  for (auto s : slice_sizes_) {
    slice_size *= s;
  }

  // Launch 2D grid of threads: indices x slice
  size_t dim0 = out.size() / slice_size;
  size_t dim1 = slice_size;
  auto group_dims = get_block_dims(dim0, dim1, 1);
  MTL::Size grid_dims = MTL::Size(dim0, dim1, 1);

  // Collect all idx shapes and strides into one place
  std::vector<int> idx_shapes;
  std::vector<size_t> idx_strides;

  for (int i = 0; i < nidx; ++i) {
    idx_shapes.insert(
        idx_shapes.end(),
        inputs[i + 1].shape().begin(),
        inputs[i + 1].shape().end());

    idx_strides.insert(
        idx_strides.end(),
        inputs[i + 1].strides().begin(),
        inputs[i + 1].strides().end());
  }

  // Set all the buffers
  compute_encoder.set_input_array(src, 0);
  compute_encoder.set_output_array(out, 1);

  // Set source info
  compute_encoder->setBytes(src.shape().data(), ndim * sizeof(int), 2);
  compute_encoder->setBytes(src.strides().data(), ndim * sizeof(size_t), 3);
  compute_encoder->setBytes(&ndim, sizeof(size_t), 4);
  compute_encoder->setBytes(slice_sizes_.data(), ndim * sizeof(int), 5);
  compute_encoder->setBytes(axes_.data(), nidx * sizeof(int), 6);

  // Set index info
  //
  // We don't need to check for empty idx_shapes because gather has a
  // idx_ndim == 0 specialization
  compute_encoder->setBytes(
      idx_shapes.data(), idx_shapes.size() * sizeof(int), 7);
  compute_encoder->setBytes(
      idx_strides.data(), idx_strides.size() * sizeof(size_t), 8);
  compute_encoder->setBytes(&idx_ndim, sizeof(int), 9);

  // Set index buffers
  for (int i = 0; i < nidx; ++i) {
    compute_encoder.set_input_array(inputs[i + 1], 20 + i);
  }

  // Launch grid
  compute_encoder.dispatchThreads(grid_dims, group_dims);
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
  auto copy_type =
      inputs[0].data_size() == 1 ? CopyType::Scalar : CopyType::General;
  copy_gpu(inputs[0], out, copy_type);

  // Empty update
  if (inputs.back().size() == 0) {
    return;
  }

  // Get stream
  auto& s = stream();
  auto& d = metal::device(s.device);

  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  bool index_nd1_specialization = (idx_ndim == 1);

  // Bail from fast path (1d index specialization) if scatter dims aren't
  // the outermost dims and contiguous since update access won't be raster
  // order.
  for (auto i = 0; i < axes_.size() && index_nd1_specialization; i++) {
    index_nd1_specialization &= (axes_[i] == i);
  }

  // Bail from fast path (1d index specialization) if any of the dims are
  // broadcasted, since we can't rely on linear indexing in that case.
  for (int i = 1; i < inputs.size() && index_nd1_specialization; i++) {
    index_nd1_specialization &= inputs[i].flags().row_contiguous;
  }

  std::string lib_name;
  std::string kernel_name;
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

  {
    std::ostringstream kname;
    if (index_nd1_specialization) {
      kname << "scatter_1d_index" << type_to_name(out) << idx_type_name;
    } else {
      kname << "scatter" << type_to_name(out) << idx_type_name;
    }
    kname << "_" << op_name << "_" << nidx;
    lib_name = kname.str();
    kernel_name = kname.str();
  }

  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::get_kernel_preamble();
    kernel_source << scatter_preamble;

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
      op_type = fmt::format(op_type, out_type_str);
    }
    auto [idx_args, idx_arr] = make_index_args(idx_type_str, nidx);

    kernel_source << fmt::format(
        scatter_kernels,
        type_to_name(out) + idx_type_name + "_" + op_name,
        out_type_str,
        idx_type_str,
        op_type,
        nidx,
        idx_args,
        idx_arr);
    lib = d.get_library(lib_name, kernel_source.str());
  }

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& upd = inputs.back();
  size_t nthreads = upd.size();

  compute_encoder->setComputePipelineState(kernel);

  // Set all the buffers
  compute_encoder.set_input_array(upd, 1);
  compute_encoder.set_output_array(out, 2);

  // Set update info
  uint upd_ndim = upd.ndim();
  size_t upd_size = 1;
  for (int i = idx_ndim; i < upd.ndim(); ++i) {
    upd_size *= upd.shape(i);
  }
  if (index_nd1_specialization) {
    compute_encoder->setBytes(
        out.shape().data(), out.shape().size() * sizeof(int), 3);
    compute_encoder->setBytes(
        out.strides().data(), out.strides().size() * sizeof(size_t), 4);
    compute_encoder->setBytes(&upd_size, sizeof(size_t), 5);

    // Set index buffers
    for (int i = 0; i < nidx; ++i) {
      compute_encoder.set_input_array(inputs[i + 1], 20 + i);
    }

    // Launch grid
    MTL::Size grid_dims = MTL::Size(upd_size, nthreads / upd_size, 1);
    MTL::Size group_dims = get_block_dims(upd_size, nthreads / upd_size, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);

  } else {
    // Collect all idx shapes and strides into one place
    std::vector<int> idx_shapes;
    std::vector<size_t> idx_strides;

    for (int i = 0; i < nidx; ++i) {
      idx_shapes.insert(
          idx_shapes.end(),
          inputs[i + 1].shape().begin(),
          inputs[i + 1].shape().end());

      idx_strides.insert(
          idx_strides.end(),
          inputs[i + 1].strides().begin(),
          inputs[i + 1].strides().end());
    }

    if (upd_ndim == 0) {
      // Need placeholders so Metal doesn't compalain
      int shape_ = 0;
      size_t stride_ = 0;
      compute_encoder->setBytes(&shape_, sizeof(int), 3);
      compute_encoder->setBytes(&stride_, sizeof(size_t), 4);
    } else {
      compute_encoder->setBytes(upd.shape().data(), upd_ndim * sizeof(int), 3);
      compute_encoder->setBytes(
          upd.strides().data(), upd_ndim * sizeof(size_t), 4);
    }
    compute_encoder->setBytes(&upd_ndim, sizeof(size_t), 5);
    compute_encoder->setBytes(&upd_size, sizeof(size_t), 6);

    // Set output info
    size_t out_ndim = out.ndim();
    if (out_ndim == 0) {
      // Need placeholders so Metal doesn't compalain
      int shape_ = 0;
      size_t stride_ = 0;
      compute_encoder->setBytes(&shape_, sizeof(int), 7);
      compute_encoder->setBytes(&stride_, sizeof(size_t), 8);
    } else {
      compute_encoder->setBytes(out.shape().data(), out_ndim * sizeof(int), 7);
      compute_encoder->setBytes(
          out.strides().data(), out_ndim * sizeof(size_t), 8);
    }
    compute_encoder->setBytes(&out_ndim, sizeof(size_t), 9);
    compute_encoder->setBytes(axes_.data(), axes_.size() * sizeof(int), 10);

    // Set index info
    if (idx_ndim == 0) {
      // Add a 0 in idx_shapes and strides to avoid the missing buffer binding
      // error in the metal API.
      idx_shapes.push_back(0);
      idx_strides.push_back(0);
    }
    compute_encoder->setBytes(
        idx_shapes.data(), idx_shapes.size() * sizeof(int), 11);
    compute_encoder->setBytes(
        idx_strides.data(), idx_strides.size() * sizeof(size_t), 12);
    compute_encoder->setBytes(&idx_ndim, sizeof(int), 13);

    // Set index buffers
    for (int i = 0; i < nidx; ++i) {
      compute_encoder.set_input_array(inputs[i + 1], 20 + i);
    }

    // Launch grid
    MTL::Size grid_dims = MTL::Size(upd_size, nthreads / upd_size, 1);
    MTL::Size group_dims = get_block_dims(upd_size, nthreads / upd_size, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }
}

} // namespace mlx::core
