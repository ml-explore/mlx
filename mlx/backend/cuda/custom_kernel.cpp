// Copyright © 2025 Apple Inc.

#include <iostream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core::fast {

namespace {

constexpr const char* default_header = R"(
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>

#define inf cuda::std::numeric_limits<float>::infinity()

)";

std::string template_arguments_hash(
    const std::vector<std::pair<std::string, TemplateArg>>& template_args) {
  if (template_args.empty()) {
    return "";
  }

  std::string hash;
  hash.reserve(512);

  for (const auto& [name, arg] : template_args) {
    if (std::holds_alternative<int>(arg)) {
      hash += fmt::format("_{}", std::get<int>(arg));
    } else if (std::holds_alternative<bool>(arg)) {
      hash += (std::get<bool>(arg)) ? "_t" : "_f";
    } else if (std::holds_alternative<Dtype>(arg)) {
      hash += "_";
      hash += get_type_string(std::get<Dtype>(arg));
    }
  }

  return hash;
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
    const std::vector<CustomKernelShapeInfo>& shape_infos) {
  std::string kernel_source;
  kernel_source.reserve(header.size() + source.size() + 8192);
  kernel_source += default_header;
  kernel_source += header;
  kernel_source +=
      "namespace mlx::core::cu {\n\n"
      "namespace cg = cooperative_groups;\n\n";

  kernel_source += "__global__ void ";
  kernel_source += func_name;
  kernel_source += "(\n";

  // Add inputs
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& name = input_names[i];
    const auto& arr = inputs[i];
    kernel_source += "    const ";
    kernel_source += dtype_to_cuda_type(arr.dtype());
    kernel_source += "* ";
    kernel_source += name;
    kernel_source += ",\n";
    // Add input shape, strides and ndim if present in the source
    if (arr.ndim() > 0) {
      if (shape_infos[i].shape) {
        kernel_source += "    const __grid_constant__ Shape ";
        kernel_source += name;
        kernel_source += "_shape,\n";
      }
      if (shape_infos[i].strides) {
        kernel_source += "    const __grid_constant__ Strides ";
        kernel_source += name;
        kernel_source += "_strides,\n";
      }
      if (shape_infos[i].ndim) {
        kernel_source += "    const __grid_constant__ int ";
        kernel_source += name;
        kernel_source += "_ndim,\n";
      }
    }
  }

  // Add outputs
  for (int i = 0; i < output_names.size(); ++i) {
    const auto& name = output_names[i];
    const auto& dtype = output_dtypes[i];
    kernel_source += "    ";
    kernel_source += dtype_to_cuda_type(dtype);
    kernel_source += "* ";
    kernel_source += name;
    if (i < output_names.size() - 1) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
  }

  // Set compile time constants
  if (!template_args.empty()) {
    for (const auto& [name, arg] : template_args) {
      if (std::holds_alternative<int>(arg)) {
        kernel_source +=
            fmt::format("  constexpr int {} = {};\n", name, std::get<int>(arg));
      } else if (std::holds_alternative<bool>(arg)) {
        kernel_source += fmt::format(
            "  constexpr bool {} = {};\n", name, std::get<bool>(arg));
      } else {
        kernel_source += fmt::format(
            "  using {} = {};\n",
            name,
            dtype_to_cuda_type(std::get<Dtype>(arg)));
      }
    }
    kernel_source += "\n";
  }

  kernel_source += source;
  kernel_source += "\n}\n\n} // namespace mlx::core::cu\n";

  return kernel_source;
}

} // namespace

CustomKernelFunction cuda_kernel(
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

  std::vector<CustomKernelShapeInfo> shape_infos;
  for (auto& n : input_names) {
    CustomKernelShapeInfo shape_info;
    shape_info.shape = source.find(n + "_shape") != std::string::npos;
    shape_info.strides = source.find(n + "_strides") != std::string::npos;
    shape_info.ndim = source.find(n + "_ndim") != std::string::npos;
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

std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory,
    std::optional<float> init_value,
    bool ensure_row_contiguous,
    StreamOrDevice s) {
  std::vector<CustomKernelShapeInfo> shape_infos(
      inputs.size(), CustomKernelShapeInfo{false, false, false});
  return array::make_arrays(
      output_shapes,
      output_dtypes,
      std::make_shared<CustomKernel>(
          to_stream(s),
          name,
          compiled_source,
          grid,
          threadgroup,
          shape_infos,
          ensure_row_contiguous,
          init_value,
          scalars,
          true,
          shared_memory),
      inputs);
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("CustomKernel::eval_gpu");
  auto& s = stream();

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
      (is_precompiled_) ? name_ : "mlx::core::cu::" + name_;
  cu::JitModule& mod = cu::get_jit_module(
      s.device,
      name_,
      [&]() {
        return std::make_tuple(
            is_precompiled_, source_, std::vector{kernel_name});
      },
      false);

  // Make the arguments
  cu::KernelArgs args;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    args.append(in);
    if (shape_info.shape) {
      args.append_ndim(in.shape());
    }
    if (shape_info.strides) {
      args.append_ndim(in.strides());
    }
    if (shape_info.ndim) {
      args.append<int32_t>(in.ndim());
    }
  }
  for (auto& out : outputs) {
    args.append(out);
  }
  for (auto& s : scalar_arguments_) {
    if (std::holds_alternative<bool>(s)) {
      args.append(std::get<bool>(s));
    } else if (std::holds_alternative<int>(s)) {
      args.append(std::get<int>(s));
    } else if (std::holds_alternative<float>(s)) {
      args.append(std::get<float>(s));
    }
  }

  // Make the grid
  const auto [tx, ty, tz] = threadgroup_;
  const auto [gx, gy, gz] = grid_;
  dim3 block(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz));
  dim3 grid((gx + tx - 1) / tx, (gy + ty - 1) / ty, (gz + tz - 1) / tz);

  // Call the kernel
  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : checked_inputs) {
    encoder.set_input_array(in);
  }
  for (const auto& out : outputs) {
    encoder.set_output_array(out);
  }
  for (const auto& t : copies) {
    encoder.add_temporary(t);
  }
  auto kernel =
      mod.get_kernel(kernel_name, [smem = shared_memory_](CUfunction kernel) {
        if (smem > 0 && smem > 48000) {
          cuFuncSetAttribute(
              kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);
        }
      });
  encoder.add_kernel_node(kernel, grid, block, shared_memory_, args.args());
}

} // namespace mlx::core::fast
