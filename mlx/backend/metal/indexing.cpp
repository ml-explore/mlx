// Copyright © 2023 Apple Inc.
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

static constexpr int METAL_MAX_INDEX_ARRAYS = 10;

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

  std::ostringstream kname;
  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";
  kname << "gather" << type_to_name(src) << idx_type_name << "_" << nidx;

  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());

  size_t slice_size = 1;
  for (auto s : slice_sizes_) {
    slice_size *= s;
  }

  size_t ndim = src.ndim();
  size_t nthreads = out.size();
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }

  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);

  compute_encoder->setComputePipelineState(kernel);

  // Make the argument buffer to store the indices for the
  // `Indices` struct in kernels/indexing.metal
  std::vector<MTL::ArgumentDescriptor*> arg_descs(4);
  arg_descs[0] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[0]->setIndex(0);
  arg_descs[0]->setDataType(MTL::DataType::DataTypePointer);
  arg_descs[0]->setArrayLength(nidx);

  // Shapes
  arg_descs[1] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[1]->setDataType(MTL::DataType::DataTypePointer);
  arg_descs[1]->setIndex(nidx + 1);

  // Strides
  arg_descs[2] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[2]->setDataType(MTL::DataType::DataTypePointer);
  arg_descs[2]->setIndex(nidx + 2);

  // Indices ndim
  arg_descs[3] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[3]->setDataType(MTL::DataType::DataTypeInt);
  arg_descs[3]->setIndex(nidx + 3);

  // Get the argument encoder
  auto arg_enc = d.argument_encoder(arg_descs);

  // Allocate and fill buffers for shapes and strides
  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  auto idx_shapes_buf = allocator::malloc_or_wait(sizeof(int) * idx_ndim);
  auto idx_strides_buf = allocator::malloc_or_wait(sizeof(size_t) * idx_ndim);
  for (int i = 0; i < nidx; ++i) {
    std::copy(
        inputs[i + 1].shape().begin(),
        inputs[i + 1].shape().end(),
        static_cast<int*>(idx_shapes_buf.raw_ptr()) + i * idx_ndim);
    std::copy(
        inputs[i + 1].strides().begin(),
        inputs[i + 1].strides().end(),
        static_cast<size_t*>(idx_strides_buf.raw_ptr()) + i * idx_ndim);
  }

  // Allocate the argument buffer
  auto arg_buf = allocator::malloc_or_wait(arg_enc->encodedLength());

  // Register data with the encoder
  arg_enc->setArgumentBuffer(static_cast<MTL::Buffer*>(arg_buf.ptr()), 0);
  for (int i = 0; i < nidx; ++i) {
    set_array_buffer(compute_encoder, arg_enc, inputs[i + 1], i);
  }
  if (idx_ndim > 0) {
    arg_enc->setBuffer(
        static_cast<MTL::Buffer*>(idx_shapes_buf.ptr()), 0, nidx + 1);
    compute_encoder->useResource(
        static_cast<MTL::Buffer*>(idx_shapes_buf.ptr()),
        MTL::ResourceUsageRead);
    arg_enc->setBuffer(
        static_cast<MTL::Buffer*>(idx_strides_buf.ptr()), 0, nidx + 2);
    compute_encoder->useResource(
        static_cast<MTL::Buffer*>(idx_strides_buf.ptr()),
        MTL::ResourceUsageRead);
  }
  *static_cast<int*>(arg_enc->constantData(nidx + 3)) = idx_ndim;

  // Set all the buffers
  set_array_buffer(compute_encoder, src, 0);
  compute_encoder->setBuffer(static_cast<MTL::Buffer*>(arg_buf.ptr()), 0, 1);
  set_array_buffer(compute_encoder, out, 2);
  compute_encoder->setBytes(src.shape().data(), ndim * sizeof(int), 3);
  compute_encoder->setBytes(src.strides().data(), ndim * sizeof(size_t), 4);
  compute_encoder->setBytes(&ndim, sizeof(size_t), 5);
  compute_encoder->setBytes(slice_sizes_.data(), ndim * sizeof(int), 6);
  compute_encoder->setBytes(&slice_size, sizeof(size_t), 7);
  compute_encoder->setBytes(axes_.data(), nidx * sizeof(int), 8);

  compute_encoder->dispatchThreads(grid_dims, group_dims);

  // Cleanup temporaries
  arg_enc->release();
  d.get_command_buffer(s.index)->addCompletedHandler(
      [arg_buf, idx_shapes_buf, idx_strides_buf](MTL::CommandBuffer*) {
        allocator::free(arg_buf);
        allocator::free(idx_shapes_buf);
        allocator::free(idx_strides_buf);
      });
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
  kname << "scatter" << type_to_name(out) << idx_type_name;
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
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }

  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);

  compute_encoder->setComputePipelineState(kernel);

  // Make the argument buffer to store the indices for the
  // `Indices` struct in kernels/indexing.metal
  std::vector<MTL::ArgumentDescriptor*> arg_descs(4);
  arg_descs[0] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[0]->setIndex(0);
  arg_descs[0]->setDataType(MTL::DataType::DataTypePointer);
  arg_descs[0]->setArrayLength(nidx);

  // Shapes
  arg_descs[1] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[1]->setDataType(MTL::DataType::DataTypePointer);
  arg_descs[1]->setIndex(nidx + 1);

  // Strides
  arg_descs[2] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[2]->setDataType(MTL::DataType::DataTypePointer);
  arg_descs[2]->setIndex(nidx + 2);

  // Indices ndim
  arg_descs[3] = MTL::ArgumentDescriptor::argumentDescriptor();
  arg_descs[3]->setDataType(MTL::DataType::DataTypeInt);
  arg_descs[3]->setIndex(nidx + 3);

  // Get the argument encoder
  auto arg_enc = d.argument_encoder(arg_descs);

  // Allocate and fill buffers for shapes and strides
  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  auto idx_shapes_buf = allocator::malloc_or_wait(sizeof(int) * idx_ndim);
  auto idx_strides_buf = allocator::malloc_or_wait(sizeof(size_t) * idx_ndim);
  for (int i = 0; i < nidx; ++i) {
    std::copy(
        inputs[i + 1].shape().begin(),
        inputs[i + 1].shape().end(),
        static_cast<int*>(idx_shapes_buf.raw_ptr()) + i * idx_ndim);
    std::copy(
        inputs[i + 1].strides().begin(),
        inputs[i + 1].strides().end(),
        static_cast<size_t*>(idx_strides_buf.raw_ptr()) + i * idx_ndim);
  }

  // Allocate the argument buffer
  auto arg_buf = allocator::malloc_or_wait(arg_enc->encodedLength());

  // Register data with the encoder
  arg_enc->setArgumentBuffer(static_cast<MTL::Buffer*>(arg_buf.ptr()), 0);
  for (int i = 0; i < nidx; ++i) {
    set_array_buffer(compute_encoder, arg_enc, inputs[i + 1], i);
  }
  if (idx_ndim > 0) {
    arg_enc->setBuffer(
        static_cast<MTL::Buffer*>(idx_shapes_buf.ptr()), 0, nidx + 1);
    compute_encoder->useResource(
        static_cast<MTL::Buffer*>(idx_shapes_buf.ptr()),
        MTL::ResourceUsageRead);
    arg_enc->setBuffer(
        static_cast<MTL::Buffer*>(idx_strides_buf.ptr()), 0, nidx + 2);
    compute_encoder->useResource(
        static_cast<MTL::Buffer*>(idx_strides_buf.ptr()),
        MTL::ResourceUsageRead);
  }
  *static_cast<int*>(arg_enc->constantData(nidx + 3)) = idx_ndim;

  compute_encoder->setBuffer(static_cast<MTL::Buffer*>(arg_buf.ptr()), 0, 0);
  size_t upd_ndim = upd.ndim();
  size_t upd_size = 1;
  for (int i = idx_ndim; i < upd.ndim(); ++i) {
    upd_size *= upd.shape(i);
  }
  set_array_buffer(compute_encoder, upd, 1);
  set_array_buffer(compute_encoder, out, 2);
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

  compute_encoder->dispatchThreads(grid_dims, group_dims);

  // Cleanup temporaries
  arg_enc->release();
  d.get_command_buffer(s.index)->addCompletedHandler(
      [arg_buf, idx_shapes_buf, idx_strides_buf](MTL::CommandBuffer*) {
        allocator::free(arg_buf);
        allocator::free(idx_shapes_buf);
        allocator::free(idx_strides_buf);
      });
}

} // namespace mlx::core
