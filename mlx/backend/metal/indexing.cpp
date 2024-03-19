// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

constexpr int METAL_MAX_INDEX_ARRAYS = 10;

} // namespace

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

  std::ostringstream kname;
  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";
  kname << "gather" << type_to_name(src) << idx_type_name << "_" << nidx;
  if (idx_ndim <= 1) {
    kname << "_" << idx_ndim;
  }

  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
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
  set_array_buffer(compute_encoder, src, 0);
  set_array_buffer(compute_encoder, out, 1);

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
  for (int i = 1; i < nidx + 1; ++i) {
    set_array_buffer(compute_encoder, inputs[i], 20 + i);
  }

  // Launch grid
  compute_encoder->dispatchThreads(grid_dims, group_dims);
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

  // Get kernel name
  std::ostringstream kname;
  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";

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

  if (index_nd1_specialization) {
    kname << "scatter_1d_index" << type_to_name(out) << idx_type_name;
  } else {
    kname << "scatter" << type_to_name(out) << idx_type_name;
  }
  switch (reduce_type_) {
    case Scatter::None:
      kname << "_none";
      break;
    case Scatter::Sum:
      kname << "_sum";
      break;
    case Scatter::Prod:
      kname << "_prod";
      break;
    case Scatter::Max:
      kname << "_max";
      break;
    case Scatter::Min:
      kname << "_min";
      break;
  }
  kname << "_" << nidx;

  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());

  auto& upd = inputs.back();
  size_t nthreads = upd.size();

  compute_encoder->setComputePipelineState(kernel);

  // Set all the buffers
  set_array_buffer(compute_encoder, upd, 1);
  set_array_buffer(compute_encoder, out, 2);

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
    for (int i = 1; i < nidx + 1; ++i) {
      set_array_buffer(compute_encoder, inputs[i], 20 + i);
    }

    // Launch grid
    MTL::Size grid_dims = MTL::Size(upd_size, nthreads / upd_size, 1);
    MTL::Size group_dims = get_block_dims(upd_size, nthreads / upd_size, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);

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
    for (int i = 1; i < nidx + 1; ++i) {
      set_array_buffer(compute_encoder, inputs[i], 20 + i);
    }

    // Launch grid
    MTL::Size grid_dims = MTL::Size(upd_size, nthreads / upd_size, 1);
    MTL::Size group_dims = get_block_dims(upd_size, nthreads / upd_size, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
}

} // namespace mlx::core
