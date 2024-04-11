// Copyright © 2023 Apple Inc.
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace mlx::core {

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  auto& in = inputs[0];

  if (axes_.size() == 0 || axes_.size() > 1 || inverse_ ||
      in.dtype() != complex64 || out.dtype() != complex64) {
    // Could also fallback to CPU implementation here.
    throw std::runtime_error(
        "GPU FFT is only implemented for 1D, forward, complex FFTs.");
  }

  size_t n = in.shape(axes_[0]);

  if (!is_power_of_2(n) || n > 2048 || n < 4) {
    throw std::runtime_error(
        "GPU FFT is only implemented for the powers of 2 from 4 -> 2048");
  }

  // Make sure that the array is contiguous and has stride 1 in the FFT dim
  std::vector<array> copies;
  auto check_input = [this, &copies, &s](const array& x) {
    // TODO: Pass the strides to the kernel so
    // we can avoid the copy when x is not contiguous.
    bool no_copy = x.strides()[axes_[0]] == 1 && x.flags().row_contiguous ||
        x.flags().col_contiguous;
    if (no_copy) {
      return x;
    } else {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      std::vector<size_t> strides;
      size_t cur_stride = x.shape(axes_[0]);
      for (int axis = 0; axis < x.ndim(); axis++) {
        if (axis == axes_[0]) {
          strides.push_back(1);
        } else {
          strides.push_back(cur_stride);
          cur_stride *= x.shape(axis);
        }
      }

      auto flags = x.flags();
      size_t f_stride = 1;
      size_t b_stride = 1;
      flags.col_contiguous = true;
      flags.row_contiguous = true;
      for (int i = 0, ri = x.ndim() - 1; i < x.ndim(); ++i, --ri) {
        flags.col_contiguous &= (strides[i] == f_stride || x.shape(i) == 1);
        f_stride *= x.shape(i);
        flags.row_contiguous &= (strides[ri] == b_stride || x.shape(ri) == 1);
        b_stride *= x.shape(ri);
      }
      // This is probably over-conservative
      flags.contiguous = false;

      x_copy.set_data(
          allocator::malloc_or_wait(x.nbytes()), x.data_size(), strides, flags);
      copy_gpu_inplace(x, x_copy, CopyType::GeneralGeneral, s);
      copies.push_back(x_copy);
      return x_copy;
    }
  };
  const array& in_contiguous = check_input(inputs[0]);

  // TODO: allow donation here
  out.set_data(
      allocator::malloc_or_wait(out.nbytes()),
      in_contiguous.data_size(),
      in_contiguous.strides(),
      in_contiguous.flags());

  // We use n / 4 threads by default since radix-4
  // is the largest single threaded radix butterfly
  // we currently implement.
  size_t m = n / 4;
  size_t batch = in.size() / in.shape(axes_[0]);

  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    std::ostringstream kname;
    kname << "fft_" << n;
    auto kernel = d.get_kernel(kname.str());

    bool donated = in.data_shared_ptr() == nullptr;
    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in_contiguous, 0);
    compute_encoder.set_output_array(out, 1);

    auto group_dims = MTL::Size(1, m, 1);
    auto grid_dims = MTL::Size(batch, m, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
