// Copyright © 2025 Apple Inc.

#include <iostream>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core::fast {

namespace {

// Inline the essential definitions for custom kernels
// This avoids the need for include paths in JIT compilation
constexpr const char* default_header = R"(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

#define inf (1.0f / 0.0f)

namespace mlx::core::rocm {

// Type aliases for convenience
using float16_t = __half;
using bfloat16_t = hip_bfloat16;

// Ceil division
template <typename T>
__host__ __device__ T ceildiv(T a, T b) {
  return (a + b - 1) / b;
}

// Thread/block index helpers
__device__ inline int thread_index() {
  return threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
}

__device__ inline int block_index() {
  return blockIdx.x + blockIdx.y * gridDim.x +
      blockIdx.z * gridDim.x * gridDim.y;
}

__device__ inline int global_thread_index() {
  return thread_index() +
      block_index() * (blockDim.x * blockDim.y * blockDim.z);
}

// Indexing helper
template <typename IdxT = int64_t>
__device__ IdxT
elem_to_loc(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

} // namespace mlx::core::rocm

)";

std::string template_arguments_hash(
    const std::vector<std::pair<std::string, TemplateArg>>& template_args) {
  if (template_args.empty()) {
    return "";
  }

  std::ostringstream hash;

  for (const auto& [name, arg] : template_args) {
    if (std::holds_alternative<int>(arg)) {
      hash << "_" << std::get<int>(arg);
    } else if (std::holds_alternative<bool>(arg)) {
      hash << (std::get<bool>(arg) ? "_t" : "_f");
    } else if (std::holds_alternative<Dtype>(arg)) {
      hash << "_" << get_type_string(std::get<Dtype>(arg));
    }
  }

  return hash.str();
}

std::string build_kernel(
    const std::string& func_name,
    const std::string& header,
    const std::string& source,
    const std::vector<std::string>& input_names,
    const std::vector<array>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<std::pair<std::string, TemplateArg>>& template_args,
    const std::vector<std::tuple<bool, bool, bool>>& shape_infos) {
  std::ostringstream kernel_source;
  kernel_source << default_header;
  kernel_source << header;
  kernel_source << "namespace mlx::core::rocm {\n\n";

  kernel_source << "__global__ void " << func_name << "(\n";

  // Add inputs
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& name = input_names[i];
    const auto& arr = inputs[i];
    kernel_source << "    const " << dtype_to_hip_type(arr.dtype()) << "* "
                  << name << ",\n";
    // Add input shape, strides and ndim if present in the source
    if (arr.ndim() > 0) {
      if (std::get<0>(shape_infos[i])) {
        kernel_source << "    const int32_t* " << name << "_shape,\n";
      }
      if (std::get<1>(shape_infos[i])) {
        kernel_source << "    const int64_t* " << name << "_strides,\n";
      }
      if (std::get<2>(shape_infos[i])) {
        kernel_source << "    const int " << name << "_ndim,\n";
      }
    }
  }

  // Add outputs
  for (size_t i = 0; i < output_names.size(); ++i) {
    const auto& name = output_names[i];
    const auto& dtype = output_dtypes[i];
    kernel_source << "    " << dtype_to_hip_type(dtype) << "* " << name;
    if (i < output_names.size() - 1) {
      kernel_source << ",\n";
    } else {
      kernel_source << ") {\n";
    }
  }

  // Set compile time constants
  if (!template_args.empty()) {
    for (const auto& [name, arg] : template_args) {
      if (std::holds_alternative<int>(arg)) {
        kernel_source << "  constexpr int " << name << " = "
                      << std::get<int>(arg) << ";\n";
      } else if (std::holds_alternative<bool>(arg)) {
        kernel_source << "  constexpr bool " << name << " = "
                      << (std::get<bool>(arg) ? "true" : "false") << ";\n";
      } else {
        kernel_source << "  using " << name << " = "
                      << dtype_to_hip_type(std::get<Dtype>(arg)) << ";\n";
      }
    }
    kernel_source << "\n";
  }

  kernel_source << source;
  kernel_source << "\n}\n\n} // namespace mlx::core::rocm\n";

  return kernel_source.str();
}

} // namespace

CustomKernelFunction hip_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    int shared_memory) {
  if (output_names.empty()) {
    throw std::invalid_argument(
        "[custom_kernel] Must specify at least one output.");
  }

  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  for (auto& n : input_names) {
    std::tuple<bool, bool, bool> shape_info;
    std::get<0>(shape_info) = source.find(n + "_shape") != std::string::npos;
    std::get<1>(shape_info) = source.find(n + "_strides") != std::string::npos;
    std::get<2>(shape_info) = source.find(n + "_ndim") != std::string::npos;
    shape_infos.push_back(shape_info);
  }

  return [=, shape_infos = std::move(shape_infos)](
             const std::vector<array>& inputs,
             const std::vector<Shape>& output_shapes,
             const std::vector<Dtype>& output_dtypes,
             std::tuple<int, int, int> grid,
             std::tuple<int, int, int> threadgroup,
             const std::vector<std::pair<std::string, TemplateArg>>&
                 template_args = {},
             std::optional<float> init_value = std::nullopt,
             bool verbose = false,
             StreamOrDevice s_ = {}) {
    if (inputs.size() != input_names.size()) {
      std::ostringstream msg;
      msg << "[custom_kernel] Expected `inputs` to have size "
          << input_names.size() << " but got size " << inputs.size() << "."
          << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_shapes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[custom_kernel] Expected `output_shapes` to have size "
          << output_names.size() << " but got size " << output_shapes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_dtypes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[custom_kernel] Expected `output_dtypes` to have size "
          << output_names.size() << " but got size " << output_dtypes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }

    auto s = to_stream(s_);
    if (s.device != Device::gpu) {
      throw std::invalid_argument("[custom_kernel] Only supports the GPU.");
    }

    std::string kernel_name =
        "custom_kernel_" + name + template_arguments_hash(template_args);
    std::string kernel_source = build_kernel(
        kernel_name,
        header,
        source,
        input_names,
        inputs,
        output_names,
        output_dtypes,
        template_args,
        shape_infos);

    if (verbose) {
      std::cout << "Generated source code for `" << kernel_name
                << "`:" << std::endl
                << "```" << std::endl
                << kernel_source << std::endl
                << "```" << std::endl;
    }

    return array::make_arrays(
        std::move(output_shapes),
        std::move(output_dtypes),
        std::make_shared<CustomKernel>(
            s,
            std::move(kernel_name),
            std::move(kernel_source),
            grid,
            threadgroup,
            shape_infos,
            ensure_row_contiguous,
            init_value,
            std::vector<ScalarArg>{},
            false,
            shared_memory),
        std::move(inputs));
  };
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  std::vector<array> copies;

  // Allocate and initialize the output arrays
  for (auto& out : outputs) {
    if (init_value_) {
      copies.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(copies.back(), out, s);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
  }

  // Create the input arrays and copy if needed
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

  // Compile the custom kernel
  std::string kernel_name =
      (is_precompiled_) ? name_ : "mlx::core::rocm::" + name_;
  rocm::JitModule& mod = rocm::get_jit_module(
      s.device,
      name_,
      [&]() {
        return std::make_tuple(
            is_precompiled_, source_, std::vector{kernel_name});
      },
      false);

  // Build argument list using KernelArgs helper
  rocm::KernelArgs args;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    args.append(in);
    if (std::get<0>(shape_info)) {
      args.append_ndim(in.shape());
    }
    if (std::get<1>(shape_info)) {
      args.append_ndim(in.strides());
    }
    if (std::get<2>(shape_info)) {
      args.append<int32_t>(in.ndim());
    }
  }
  for (auto& out : outputs) {
    args.append(out);
  }

  // Make the grid
  const auto [tx, ty, tz] = threadgroup_;
  const auto [gx, gy, gz] = grid_;
  dim3 block(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz));
  dim3 grid((gx + tx - 1) / tx, (gy + ty - 1) / ty, (gz + tz - 1) / tz);

  // Set up arrays for kernel
  for (const auto& in : checked_inputs) {
    encoder.set_input_array(in);
  }
  for (const auto& out : outputs) {
    encoder.set_output_array(out);
  }
  for (const auto& t : copies) {
    encoder.add_temporary(t);
  }

  // Launch kernel
  encoder.launch_kernel([&](hipStream_t stream) {
    auto kernel = mod.get_kernel(kernel_name);

    (void)hipModuleLaunchKernel(
        kernel,
        grid.x,
        grid.y,
        grid.z,
        block.x,
        block.y,
        block.z,
        shared_memory_,
        stream,
        args.args(),
        nullptr);
  });
}

} // namespace mlx::core::fast
