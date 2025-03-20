// Copyright © 2023-2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int SOFTMAX_LOOPED_LIMIT = 4096;

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[softmax] Does not support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Make sure that the last dimension is contiguous
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array in = set_output(inputs[0]);

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = SOFTMAX_N_READS;
  const int looped_limit = SOFTMAX_LOOPED_LIMIT;

  std::string kernel_name = (axis_size > looped_limit) ? "looped_" : "block_";
  kernel_name += "softmax_";
  if (in.dtype() != float32 && precise_) {
    kernel_name += "precise_";
  }
  kernel_name += type_to_name(out);

  auto kernel = get_softmax_kernel(d, kernel_name, precise_, out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
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

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(axis_size, 2);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

} // namespace mlx::core
