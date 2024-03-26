// Copyright © 2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
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

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    }

    array x_copy(x.shape(), x.dtype(), nullptr, {});
    copy_gpu(x, x_copy, CopyType::General, s);
    copies.push_back(x_copy);
    return x_copy;
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];
  const array& g = check_input(inputs[2]);
  array& gx = outputs[0];
  array& gw = outputs[1];

  // Allocate space for the outputs
  bool x_in_gx = false;
  bool g_in_gx = false;
  if (x.is_donatable()) {
    gx.move_shared_buffer(x);
    x_in_gx = true;
  } else if (g.is_donatable()) {
    gx.move_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc_or_wait(gx.nbytes()));
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate a temporary to store the gradients for w and initialize the
  // gradient accumulator to 0.
  array gw_temp({n_rows, x.shape().back()}, gw.dtype(), nullptr, {});
  bool g_in_gw = false;
  if (!g_in_gx && g.is_donatable()) {
    gw_temp.move_shared_buffer(g);
    g_in_gw = true;
  } else {
    gw_temp.set_data(allocator::malloc_or_wait(gw_temp.nbytes()));
  }
  copies.push_back(gw_temp);
  array zero(0, gw.dtype());
  copy_gpu(zero, gw, CopyType::Scalar, s);

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "vjp_rms";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(gx);
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
    set_array_buffer(compute_encoder, x_in_gx ? gx : x, 0);
    set_array_buffer(compute_encoder, w, 1);
    set_array_buffer(
        compute_encoder, g_in_gx ? gx : (g_in_gw ? gw_temp : g), 2);
    set_array_buffer(compute_encoder, gx, 3);
    set_array_buffer(compute_encoder, gw_temp, 4);
    compute_encoder->setBytes(&eps_, sizeof(float), 5);
    compute_encoder->setBytes(&axis_size, sizeof(int), 6);
    compute_encoder->setBytes(&w_stride, sizeof(uint32_t), 7);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }

  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
  strided_reduce_general_dispatch(
      gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);

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

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    }

    array x_copy(x.shape(), x.dtype(), nullptr, {});
    copy_gpu(x, x_copy, CopyType::General, s);
    copies.push_back(x_copy);
    return x_copy;
  };
  const array& x = check_input(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];
  const array& g = check_input(inputs[3]);
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

  // Allocate space for the outputs
  bool x_in_gx = false;
  bool g_in_gx = false;
  if (x.is_donatable()) {
    gx.move_shared_buffer(x);
    x_in_gx = true;
  } else if (g.is_donatable()) {
    gx.move_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc_or_wait(gx.nbytes()));
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate a temporary to store the gradients for w and initialize the
  // gradient accumulator to 0.
  array gw_temp({n_rows, x.shape().back()}, gw.dtype(), nullptr, {});
  bool g_in_gw = false;
  if (!g_in_gx && g.is_donatable()) {
    gw_temp.move_shared_buffer(g);
    g_in_gw = true;
  } else {
    gw_temp.set_data(allocator::malloc_or_wait(gw_temp.nbytes()));
  }
  copies.push_back(gw_temp);
  array zero(0, gw.dtype());
  copy_gpu(zero, gw, CopyType::Scalar, s);
  copy_gpu(zero, gb, CopyType::Scalar, s);

  // Finish with the gradient for b in case we had a b
  auto compute_encoder = d.get_command_encoder(s.index);
  if (gb.ndim() == 1 && gb.size() == axis_size) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        g, gb, "sum", plan, {0}, compute_encoder, d, s);
  }

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "vjp_layer_norm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(gx);
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
    set_array_buffer(compute_encoder, x_in_gx ? gx : x, 0);
    set_array_buffer(compute_encoder, w, 1);
    set_array_buffer(
        compute_encoder, g_in_gx ? gx : (g_in_gw ? gw_temp : g), 2);
    set_array_buffer(compute_encoder, gx, 3);
    set_array_buffer(compute_encoder, gw_temp, 4);
    compute_encoder->setBytes(&eps_, sizeof(float), 5);
    compute_encoder->setBytes(&axis_size, sizeof(int), 6);
    compute_encoder->setBytes(&w_stride, sizeof(uint32_t), 7);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }

  if (gw.ndim() == 1 && gw.size() == axis_size) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);
  }

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core::fast
