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
// Threadgroup memory batching improves throughput for small n
#define MIN_THREADGROUP_MEM_SIZE 64

std::pair<int, std::vector<int>> FFT::next_fast_n(int n) {
  while (n <= MAX_SINGLE_FFT_SIZE) {
    // Powers of 2 are so fast that it's worth skipping the composites
    if (is_power_of_2(n)) {
      auto plan = plan_stockham_fft(n);
      return std::make_pair(n, plan);
    }
    n += 1;
  }
  throw std::runtime_error(
      "Next fast FFT size is larger than the maximum that fits in shared memory.");
}

// Plan the sequence of radices
std::vector<int> FFT::plan_stockham_fft(int n) {
  // prefer larger radices since we do fewer expensive twiddles
  const std::vector<int> supported_radices = {13, 11, 7, 5, 4, 3, 2};
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
  // return an empty vector if unplannable
  return std::vector<int>();
}

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  auto& in = inputs[0];

  if (axes_.size() == 0) {
    throw std::runtime_error("GPU FFT is not implemented for 0D transforms.");
  }

  size_t n = out.dtype() == float32 ? out.shape(axes_[0]) : in.shape(axes_[0]);

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

  int bluestein_n = -1;
  auto plan = plan_stockham_fft(n);
  // for (int p: plan) {
  //   std::cout << "plan " << p << std::endl;
  // }
  if (plan.size() == 0) {
    // Bluestein's algorithm transforms an FFT to
    // a convolution of size > 2n + 1.
    // We solve that conv via FFT wth the convolution theorem.
    std::tie(bluestein_n, plan) = next_fast_n(2 * n - 1);
  }

  int fft_size = bluestein_n > 0 ? bluestein_n : n;

  auto make_int = [](int* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeInt, i);
  };
  auto make_bool = [](bool* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeBool, i);
  };
  // Pass this as a function constant so we can specialize
  // the kernel for powers of 2.
  bool power_of_2 = is_power_of_2(fft_size);

  std::vector<MTLFC> func_consts = {
      make_bool(&inverse_, 0), make_bool(&power_of_2, 1)};

  const std::vector<int> supported_radices = {13, 11, 7, 5, 4, 3, 2};
  int index = 3;
  int elems_per_thread = 0;
  for (int i = 0; i < plan.size(); i++) {
    func_consts.push_back(make_int(&plan[i], index));
    index += 1;
    if (plan[i] > 0) {
      elems_per_thread = std::max(elems_per_thread, supported_radices[i]);
    }
  }
  func_consts.push_back(make_int(&elems_per_thread, 2));

  // The overall number of FFTs we're going to compute for this input
  int total_batch_size = in.size() / in.shape(axes_[0]);

  int threads_per_fft = fft_size / elems_per_thread;

  // We batch among threadgroups for improved efficiency when n is small
  int threadgroup_batch_size = std::max(MIN_THREADGROUP_MEM_SIZE / fft_size, 1);
  int threadgroup_mem_size = next_power_of_2(threadgroup_batch_size * fft_size);
  // std::cout << "threadgroup_mem_size " << threadgroup_mem_size << std::endl;

  // ceil divide
  int batch_size =
      (total_batch_size + threadgroup_batch_size - 1) / threadgroup_batch_size;

  // std::cout << "batch_size " << batch_size << std::endl;
  // std::cout << "threadgroup_batch_size " << threadgroup_batch_size <<
  // std::endl; std::endl; std::cout << "threads_per_fft " << threads_per_fft <<
  // std::endl; std::cout << "total_batch_size " << total_batch_size <<
  // std::endl; std::cout << "n " << n << std::endl; std::cout << "bluestein_n "
  // << bluestein_n << std::endl; std::cout << "elems_per_thread " <<
  // elems_per_thread << std::endl;

  int out_buffer_size = out.size();

  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    std::ostringstream kname;
    std::string inv_string = inverse_ ? "true" : "false";
    if (bluestein_n > 0) {
      kname << "bluestein_fft_mem_" << threadgroup_mem_size;
    } else if (out.dtype() == float32) {
      kname << "irfft_mem_" << threadgroup_mem_size;
    } else if (in.dtype() == float32) {
      kname << "rfft_mem_" << threadgroup_mem_size;
    } else {
      kname << "fft_mem_" << threadgroup_mem_size;
    }
    std::string base_name = kname.str();
    // We use a specialized kernel for each FFT size
    kname << "_n_" << fft_size << "_inv_" << inverse_;
    std::string hash_name = kname.str();
    // std::cout << "hash_name " << hash_name << std::endl;
    auto kernel = d.get_kernel(base_name, "mlx", hash_name, func_consts);

    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in_contiguous, 0);
    compute_encoder.set_output_array(out, 1);

    if (bluestein_n > 0) {
      // Precomputed twiddle factors for Bluestein's
      auto& w_q = inputs[1];
      auto& w_k = inputs[2];
      compute_encoder.set_input_array(w_q, 2); // w_q
      compute_encoder.set_input_array(w_k, 3); // w_k
      compute_encoder->setBytes(&n, sizeof(int), 4);
      compute_encoder->setBytes(&bluestein_n, sizeof(int), 5);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 6);
    } else {
      compute_encoder->setBytes(&n, sizeof(int), 2);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 3);
    }

    // std::cout << "input shape " << in.shape(0) << " " << in.shape(1) <<
    // std::endl;
    // std::cout << "out shape " << out.shape(0) << " " << out.shape(1) <<
    // std::endl;

    auto group_dims = MTL::Size(1, threadgroup_batch_size, threads_per_fft);
    auto grid_dims =
        MTL::Size(batch_size, threadgroup_batch_size, threads_per_fft);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
