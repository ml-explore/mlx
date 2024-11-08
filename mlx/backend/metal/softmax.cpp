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
  std::vector<array> copies;
  auto check_input = [&copies, &s](const array& x) -> const array& {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
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
    compute_encoder.set_input_array(
        in.data_shared_ptr() == nullptr ? out : in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(axis_size, 2);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core
