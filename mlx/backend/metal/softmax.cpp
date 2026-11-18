// Copyright © 2023-2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
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
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
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
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array in = set_output(inputs[0]);

  int axis_size = in.shape().back();
  int64_t n_rows = in.data_size() / axis_size;

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
  auto& compute_encoder = metal::get_command_encoder(s);
  {
    size_t threadgroup_size;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
    } else {
      threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
    }

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_bytes(axis_size, 2);
    const int64_t row_bytes = int64_t(axis_size) * out.itemsize();
    dispatch_threadgroups_in_slices(
        compute_encoder, n_rows, threadgroup_size, [&](int64_t row) {
          compute_encoder.set_input_array(in, 0, row * row_bytes);
          compute_encoder.set_output_array(out, 1, row * row_bytes);
        });
  }
}

} // namespace mlx::core
