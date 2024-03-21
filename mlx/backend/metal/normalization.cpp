// Copyright © 2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) {
    bool no_copy = x.strides()[x.ndim() - 1] == 1;
    if (x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      return x;
    } else {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      copies.push_back(x_copy);
      return x_copy;
    }
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];

  if (x.is_donatable()) {
    out.move_shared_buffer(x);
  } else {
    out.set_data(
        allocator::malloc_or_wait(x.data_size() * x.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "rms";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(out);
  auto compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = w.strides()[0];
    compute_encoder->setComputePipelineState(kernel);
    set_array_buffer(
        compute_encoder, x.data_shared_ptr() == nullptr ? out : x, 0);
    set_array_buffer(compute_encoder, w, 1);
    set_array_buffer(compute_encoder, out, 2);
    compute_encoder->setBytes(&eps_, sizeof(float), 3);
    compute_encoder->setBytes(&axis_size, sizeof(int), 4);
    compute_encoder->setBytes(&w_stride, sizeof(uint32_t), 5);
    compute_encoder->setThreadgroupMemoryLength(
        16 * 8, 0); // minimum of 16 bytes
    compute_encoder->setThreadgroupMemoryLength(simd_size * sizeof(float), 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) {
    bool no_copy = x.strides()[x.ndim() - 1] == 1;
    if (x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      return x;
    } else {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      copies.push_back(x_copy);
      return x_copy;
    }
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];

  if (x.is_donatable()) {
    out.move_shared_buffer(x);
  } else {
    out.set_data(
        allocator::malloc_or_wait(x.data_size() * x.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "layer_norm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(out);
  auto compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    uint32_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;
    compute_encoder->setComputePipelineState(kernel);
    set_array_buffer(
        compute_encoder, x.data_shared_ptr() == nullptr ? out : x, 0);
    set_array_buffer(compute_encoder, w, 1);
    set_array_buffer(compute_encoder, b, 2);
    set_array_buffer(compute_encoder, out, 3);
    compute_encoder->setBytes(&eps_, sizeof(float), 4);
    compute_encoder->setBytes(&axis_size, sizeof(int), 5);
    compute_encoder->setBytes(&w_stride, sizeof(uint32_t), 6);
    compute_encoder->setBytes(&b_stride, sizeof(uint32_t), 7);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core::fast
