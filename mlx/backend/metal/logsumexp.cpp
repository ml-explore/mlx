// Copyright © 2023-2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int LOGSUMEXP_LOOPED_LIMIT = 4096;

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[logsumexp] Does not support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  // Make sure that the last dimension is contiguous
  auto ensure_contiguous = [&](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      compute_encoder.add_temporary(x_copy);
      return x_copy;
    }
  };

  auto in = ensure_contiguous(inputs[0]);
  if (in.flags().row_contiguous) {
    out.set_data(allocator::malloc(out.nbytes()));
  } else {
    auto n = in.shape(-1);
    auto flags = in.flags();
    auto strides = in.strides();
    for (auto& s : strides) {
      s /= n;
    }
    bool col_contig = strides[0] == 1;
    for (int i = 1; col_contig && i < strides.size(); ++i) {
      col_contig &=
          (out.shape(i) == 1 || strides[i - 1] == out.shape(i) * strides[i]);
    }
    flags.col_contiguous = col_contig;
    out.set_data(
        allocator::malloc(in.nbytes() / n),
        in.data_size() / n,
        std::move(strides),
        flags);
  }

  int axis_size = in.shape().back();
  int64_t n_rows = in.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = 4;
  const int looped_limit = LOGSUMEXP_LOOPED_LIMIT;

  std::string kernel_name = (axis_size > looped_limit) ? "looped_" : "block_";
  kernel_name += "logsumexp_";
  kernel_name += type_to_name(out);

  auto kernel = get_logsumexp_kernel(d, kernel_name, out);
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
    const int64_t in_row_bytes = int64_t(axis_size) * in.itemsize();
    dispatch_threadgroups_in_slices(
        compute_encoder, n_rows, threadgroup_size, [&](int64_t row) {
          compute_encoder.set_input_array(in, 0, row * in_row_bytes);
          compute_encoder.set_output_array(out, 1, row * out.itemsize());
        });
  }
}

} // namespace mlx::core
