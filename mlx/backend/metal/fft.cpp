// Copyright © 2023 Apple Inc.
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <iostream>

namespace mlx::core {

using MTLFC = std::tuple<const void*, MTL::DataType, NS::UInteger>;

#define MAX_SINGLE_FFT_SIZE 2048

bool is_fast(int n, const std::vector<int>& supported_radices) {
  for (auto radix : supported_radices) {
    if (n % radix == 0) {
      return true;
    }
  }
  return false;
}

int next_fast_n(int n, const std::vector<int>& supported_radices) {
  while (n < MAX_SINGLE_FFT_SIZE) {
    if (is_fast(n, supported_radices)) {
      return n;
    }
    n += 1;
  }
  return MAX_SINGLE_FFT_SIZE;
}

// Plan the sequence of radices
std::vector<int> plan_stockham_fft(
    int n,
    const std::vector<int>& supported_radices) {
  // prefer larger radices since we do fewer expensive twiddles
  std::vector<int> plan(supported_radices.size());
  for (int i = 0; i < supported_radices.size(); i++) {
    int radix = supported_radices[i];
    while (n % radix == 0) {
      plan[i] += 1;
      n /= radix;
      if (n == 1) {
        return plan;
      }
    }
  }
  throw std::runtime_error(
      "n should be decomposable into the supported radices.");
}

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  auto& in = inputs[0];

  if (axes_.size() == 0) {
    throw std::runtime_error("GPU FFT is not implemented for 0D transforms.");
  }

  int n = out.dtype() == float32 ? out.shape(axes_[0]) : in.shape(axes_[0]);

  if (n > MAX_SINGLE_FFT_SIZE || n < 3) {
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

  const std::vector<int> supported_radices = {4, 3, 2};

  // Bluestein's algorithm transforms to an FFT of
  // the first power of 2 after (2 * n + 1)
  int bluestein_n = next_fast_n(2 * n - 1, supported_radices);

  int fft_size = is_fast(n, supported_radices) ? n : bluestein_n;

  size_t batch = in.size() / in.shape(axes_[0]);

  auto make_int = [](int* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeInt, i);
  };
  auto make_bool = [](bool* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeBool, i);
  };

  std::vector<MTLFC> func_consts = {make_bool(&inverse_, 0)};

  auto plan = plan_stockham_fft(n, supported_radices);
  int index = 2;
  // based on the max radix size used
  int elems_per_thread = 0;
  for (int i = 0; i < plan.size(); i++) {
    func_consts.push_back(make_int(&plan[i], index));
    index += 1;
    if (plan[i] > 0) {
      elems_per_thread = std::max(elems_per_thread, supported_radices[i]);
    }
  }
  func_consts.push_back(make_int(&elems_per_thread, 1));

  int threads_per_fft = fft_size / elems_per_thread;
  int threadgroup_mem_size = next_power_of_2(n);

  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    std::ostringstream kname;
    std::string inv_string = inverse_ ? "true" : "false";
    if (!is_fast(n, supported_radices)) {
      kname << "bluestein_fft_" << next_power_of_2(bluestein_n);
    } else if (out.dtype() == float32) {
      kname << "irfft_" << threadgroup_mem_size;
    } else if (in.dtype() == float32) {
      kname << "rfft_" << threadgroup_mem_size;
    } else {
      kname << "fft_" << threadgroup_mem_size;
    }
    auto kernel = d.get_kernel(kname.str(), "mlx", "", func_consts);

    bool donated = in.data_shared_ptr() == nullptr;
    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in_contiguous, 0);
    compute_encoder.set_output_array(out, 1);
    if (!is_fast(n, supported_radices)) {
      // Bluestein requires extra pre-computed inputs
      auto& w_q = inputs[1];
      auto& w_k = inputs[2];
      set_array_buffer(compute_encoder, w_q, 2);
      set_array_buffer(compute_encoder, w_k, 3);
      compute_encoder->setBytes(&n, sizeof(int), 4);
      compute_encoder->setBytes(&bluestein_n, sizeof(int), 5);
    } else {
      compute_encoder->setBytes(&n, sizeof(int), 2);
    }

    auto group_dims = MTL::Size(1, threads_per_fft, 1);
    auto grid_dims = MTL::Size(batch, threads_per_fft, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
