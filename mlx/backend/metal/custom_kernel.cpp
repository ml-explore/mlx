// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

struct CustomKernelCache {
  std::unordered_map<std::string, std::string> libraries;
};

static CustomKernelCache& cache() {
  static CustomKernelCache cache_;
  return cache_;
};

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // silence some warnings
  (void)is_precompiled_;
  (void)shared_memory_;

  auto& s = stream();

  std::vector<array> copies;

  for (auto& out : outputs) {
    if (init_value_) {
      copies.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(copies.back(), out, s);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
  }

  auto check_input = [&copies, &s, this](const array& x) -> const array {
    bool no_copy = x.flags().row_contiguous;
    if (!ensure_row_contiguous_ || no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  std::vector<array> checked_inputs;
  for (const array& in : inputs) {
    checked_inputs.push_back(check_input(in));
  }

  auto& d = metal::device(s.device);

  {
    // Clear kernels from the device library cache if needed
    auto& kernel_cache = cache();
    if (auto it = kernel_cache.libraries.find(name_);
        it != kernel_cache.libraries.end()) {
      if (it->second != source_) {
        auto& d = metal::device(s.device);
        d.clear_library(name_);
        it->second = source_;
      }
    } else {
      kernel_cache.libraries.emplace(name_, source_);
    }
  }

  auto lib = d.get_library(name_, [this] { return metal::utils() + source_; });
  auto kernel = d.get_kernel(name_, lib);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  int index = 0;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    compute_encoder.set_input_array(in, index);
    index++;
    if (in.ndim() > 0) {
      int ndim = in.ndim();
      if (std::get<0>(shape_info)) {
        compute_encoder.set_vector_bytes(in.shape(), ndim, index);
        index++;
      }
      if (std::get<1>(shape_info)) {
        compute_encoder.set_vector_bytes(in.strides(), ndim, index);
        index++;
      }
      if (std::get<2>(shape_info)) {
        compute_encoder.set_bytes(ndim, index);
        index++;
      }
    }
  }
  for (auto& out : outputs) {
    compute_encoder.set_output_array(out, index);
    index++;
  }

  const auto [tx, ty, tz] = threadgroup_;
  auto tg_size = tx * ty * tz;
  auto max_tg_size = kernel->maxTotalThreadsPerThreadgroup();
  if (tg_size > max_tg_size) {
    std::ostringstream msg;
    msg << "Thread group size (" << tg_size << ") is greater than "
        << " the maximum allowed threads per threadgroup (" << max_tg_size
        << ").";
    throw std::invalid_argument(msg.str());
  }

  const auto [gx, gy, gz] = grid_;
  MTL::Size group_dims =
      MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz));
  MTL::Size grid_dims = MTL::Size(gx, gy, gz);
  compute_encoder.dispatch_threads(grid_dims, group_dims);

  compute_encoder.add_temporaries(std::move(copies));
}

} // namespace mlx::core::fast
