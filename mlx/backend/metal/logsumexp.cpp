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

  // Make sure that the last dimension is contiguous
  auto ensure_contiguous = [&s, &d](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      d.add_temporary(x_copy, s.index);
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
  int n_rows = in.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = 4;
  const int looped_limit = LOGSUMEXP_LOOPED_LIMIT;

  std::string kernel_name = (axis_size > looped_limit) ? "looped_" : "block_";
  kernel_name += "logsumexp_";
  kernel_name += type_to_name(out);

  auto kernel = get_logsumexp_kernel(d, kernel_name, out);
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
