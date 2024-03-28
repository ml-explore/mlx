// Copyright © 2023 Apple Inc.

#include <algorithm>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[softmax] Does not support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

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
  const array& in = check_input(inputs[0]);
  if (in.is_donatable()) {
    out.move_shared_buffer(in);
  } else {
    out.set_data(
        allocator::malloc_or_wait(in.data_size() * in.itemsize()),
        in.data_size(),
        in.strides(),
        in.flags());
  }

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = SOFTMAX_N_READS;
  const int looped_limit = SOFTMAX_LOOPED_LIMIT;
  std::string op_name = "softmax_";
  if (axis_size > looped_limit) {
    op_name += "looped_";
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

    compute_encoder->setComputePipelineState(kernel);
    set_array_buffer(
        compute_encoder, in.data_shared_ptr() == nullptr ? out : in, 0);
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
    compute_encoder->setBytes(&axis_size, sizeof(int), 2);
    compute_encoder->setThreadgroupMemoryLength(simd_size * in.itemsize(), 0);
    compute_encoder->setThreadgroupMemoryLength(simd_size * in.itemsize(), 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
