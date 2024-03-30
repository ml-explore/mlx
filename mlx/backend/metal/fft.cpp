// Copyright © 2023 Apple Inc.
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <iostream>

namespace mlx::core {

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  auto& in = inputs[0];

  if (axes_.size() == 0) {
    throw std::runtime_error("GPU FFT is not implemented for 0D transforms.");
  }

  int n = out.dtype() == float32 ? out.shape(axes_[0]) : in.shape(axes_[0]);

  if (n > 2048 || n < 4) {
    throw std::runtime_error("GPU FFT is only implemented from 3 -> 2048");
  }

  // Make sure that the array is contiguous and has stride 1 in the FFT dim
  std::vector<array> copies;
  auto check_input = [this, &copies, &s](const array& x) -> const array& {
    // TODO: Pass the strides to the kernel so
    // we can avoid the copy when x is not contiguous.
    bool no_copy = x.strides()[axes_[0]] == 1 && x.flags().row_contiguous ||
        x.flags().col_contiguous;
    if (no_copy) {
      return x;
    } else {
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

      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copies.back().set_data(
          allocator::malloc_or_wait(x.nbytes()), x.data_size(), strides, flags);
      copy_gpu_inplace(x, copies.back(), CopyType::GeneralGeneral, s);
      return copies.back();
    }
  };
  const array& in_contiguous = check_input(inputs[0]);

  // real to complex: n -> (n/2)+1
  // complex to real: (n/2)+1 -> n
  auto out_strides = in_contiguous.strides();
  if (in.dtype() != out.dtype()) {
    for (int i = 0; i < out_strides.size(); i++) {
      if (out_strides[i] != 1) {
        out_strides[i] =
            out_strides[i] / in.shape(axes_[0]) * out.shape(axes_[0]);
      }
    }
  }
  // TODO: allow donation here
  out.set_data(
      allocator::malloc_or_wait(out.nbytes()),
      out.data_size(),
      out_strides,
      in_contiguous.flags());

  size_t batch = in.size() / in.shape(axes_[0]);
  // Bluestein's algorithm transforms to an FFT of
  // the first power of 2 after (2 * n + 1)
  int bluestein_n = next_power_of_2(2 * n - 1);

  int m = is_power_of_2(n) ? n : bluestein_n;

  // We use n / 4 threads by default since radix-4
  // is the largest single threaded radix butterfly
  // we currently implement.
  size_t m = n / 4;
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    std::ostringstream kname;
    if (!is_power_of_2(n)) {
      kname << "bluestein_" << bluestein_n;
    } else if (out.dtype() == float32) {
      kname << "irfft_" << n;
    } else if (in.dtype() == float32) {
      kname << "rfft_" << n;
    } else {
      kname << "fft_" << n << "_inv_";
      if (inverse_) {
        kname << "true";
      } else {
        kname << "false";
      }
    }
    auto kernel = d.get_kernel(kname.str());

    bool donated = in.data_shared_ptr() == nullptr;
    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in_contiguous, 0);
    compute_encoder.set_output_array(out, 1);
    if (!is_power_of_2(n)) {
      // Bluestein requires extra pre-computed inputs
      auto& w_q = inputs[1];
      auto& w_k = inputs[2];
      set_array_buffer(compute_encoder, w_q, 2);
      set_array_buffer(compute_encoder, w_k, 3);
      compute_encoder->setBytes(&n, sizeof(int), 4);
    }

    auto group_dims = MTL::Size(1, m, 1);
    auto grid_dims = MTL::Size(batch, m, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
