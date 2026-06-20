// Copyright © 2024 Apple Inc.

#include <iostream>
#include <regex>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

// Inputs with fewer elements are passed in the constant address space
constexpr int max_constant_array_size = 8;

Stream resolve_metal_kernel_stream(StreamOrDevice s) {
  if (metal::is_available()) {
    // Default an unspecified stream to the GPU rather than the global default
    // device. Other inputs go through to_stream so explicit streams, devices
    // and ThreadLocalStream are handled as before.
    auto stream = std::holds_alternative<std::monostate>(s)
        ? default_stream(Device::gpu)
        : to_stream(s);
    if (stream.device != Device::gpu) {
      throw std::invalid_argument("[metal_kernel] Only supports the GPU.");
    }
    return stream;
  }
  if (!detail::in_export_tracing()) {
    throw std::runtime_error("[metal_kernel] No Metal back-end.");
  }
  // Exporting without Metal: the kernel cannot run here but it can still be
  // recorded in the graph on a placeholder GPU stream. The importing process
  // remaps it to one of its own streams.
  auto* device = std::get_if<Device>(&s);
  auto* stream = std::get_if<Stream>(&s);
  auto* tl_stream = std::get_if<ThreadLocalStream>(&s);
  if ((device && *device != Device::gpu) ||
      (stream && stream->device != Device::gpu) ||
      (tl_stream && tl_stream->device != Device::gpu)) {
    throw std::invalid_argument("[metal_kernel] Only supports the GPU.");
  }
  return Stream(-1, Device::gpu);
}

std::string write_signature(
    std::string func_name,
    const std::string& header,
    const std::string& source,
    const std::vector<std::string>& input_names,
    const std::vector<array>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<std::pair<std::string, TemplateArg>>& template_args,
    const std::vector<std::string>& attributes,
    const std::vector<std::tuple<bool, bool, bool>>& shape_infos,
    bool atomic_outputs) {
  std::string kernel_source;
  kernel_source.reserve(header.size() + source.size() + 16384);
  kernel_source += header;
  // Auto-generate a function signature based on `template_args`
  // and the dtype/shape of the arrays passed as `inputs`.
  if (!template_args.empty()) {
    kernel_source += "template <";
    int i = 0;
    for (const auto& [name, arg] : template_args) {
      std::string param_type;
      if (std::holds_alternative<int>(arg)) {
        param_type = "int";
      } else if (std::holds_alternative<bool>(arg)) {
        param_type = "bool";
      } else if (std::holds_alternative<Dtype>(arg)) {
        param_type = "typename";
      }
      if (i > 0) {
        kernel_source += ", ";
      }
      kernel_source += param_type;
      kernel_source += " ";
      kernel_source += name;
      i++;
    }
    kernel_source += ">\n";
  }
  kernel_source += "[[kernel]] void ";
  kernel_source += func_name;
  kernel_source += "(\n";

  int index = 0;
  // Add inputs
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& name = input_names[i];
    const auto& arr = inputs[i];
    auto dtype = get_type_string(arr.dtype());
    std::string location =
        arr.size() < max_constant_array_size ? "constant" : "device";
    std::string ref = arr.ndim() == 0 ? "&" : "*";
    kernel_source += "  const ";
    kernel_source += location;
    kernel_source += " ";
    kernel_source += dtype;
    kernel_source += ref;
    kernel_source += " ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]],\n";
    index++;
    // Add input shape, strides and ndim if present in the source
    if (arr.ndim() > 0) {
      if (std::get<0>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int* " + name + "_shape [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (std::get<1>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int64_t* " + name + "_strides [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (std::get<2>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int& " + name + "_ndim [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
    }
  }
  // Add outputs
  for (int i = 0; i < output_names.size(); ++i) {
    const auto& name = output_names[i];
    const auto& dtype = output_dtypes[i];
    kernel_source += "  device ";
    auto type_string = get_type_string(dtype);
    if (atomic_outputs) {
      kernel_source += "atomic<";
    }
    kernel_source += type_string;
    if (atomic_outputs) {
      kernel_source += ">";
    }
    kernel_source += "* ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]]";
    if (index < inputs.size() + output_names.size() - 1 ||
        attributes.size() > 0) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }

  index = 0;
  for (const auto& attr : attributes) {
    kernel_source += attr;
    if (index < attributes.size() - 1) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }
  kernel_source += source;
  kernel_source += "\n}\n";
  return kernel_source;
}

std::string write_template(
    const std::vector<std::pair<std::string, TemplateArg>>& template_args) {
  std::ostringstream template_def;
  template_def << "<";
  int i = 0;
  for (const auto& [name, arg] : template_args) {
    if (i > 0) {
      template_def << ", ";
    }
    if (std::holds_alternative<int>(arg)) {
      template_def << std::get<int>(arg);
    } else if (std::holds_alternative<bool>(arg)) {
      template_def << std::get<bool>(arg);
    } else if (std::holds_alternative<Dtype>(arg)) {
      template_def << get_type_string(std::get<Dtype>(arg));
    }
    i++;
  }
  template_def << ">";
  return template_def.str();
}

} // namespace

CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header /* = "" */,
    bool ensure_row_contiguous /* = true */,
    bool atomic_outputs /* = false */,
    CompileOptions compile_options /* = {} */) {
  if (output_names.empty()) {
    throw std::invalid_argument(
        "[metal_kernel] Must specify at least one output.");
  }
  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  for (auto& n : input_names) {
    std::tuple<bool, bool, bool> shape_info;
    std::get<0>(shape_info) = source.find(n + "_shape") != std::string::npos;
    std::get<1>(shape_info) = source.find(n + "_strides") != std::string::npos;
    std::get<2>(shape_info) = source.find(n + "_ndim") != std::string::npos;
    shape_infos.push_back(shape_info);
  }
  const std::vector<std::pair<std::string, std::string>> metal_attributes = {
      {"dispatch_quadgroups_per_threadgroup", "uint"},
      {"dispatch_simdgroups_per_threadgroup", "uint"},
      {"dispatch_threads_per_threadgroup", "uint3"},
      {"grid_origin", "uint3"},
      {"grid_size", "uint3"},
      {"quadgroup_index_in_threadgroup", "uint"},
      {"quadgroups_per_threadgroup", "uint"},
      {"simdgroup_index_in_threadgroup", "uint"},
      {"simdgroups_per_threadgroup", "uint"},
      {"thread_execution_width", "uint"},
      {"thread_index_in_quadgroup", "uint"},
      {"thread_index_in_simdgroup", "uint"},
      {"thread_index_in_threadgroup", "uint"},
      {"thread_position_in_grid", "uint3"},
      {"thread_position_in_threadgroup", "uint3"},
      {"threadgroup_position_in_grid", "uint3"},
      {"threadgroups_per_grid", "uint3"},
      {"threads_per_grid", "uint3"},
      {"threads_per_simdgroup", "uint"},
      {"threads_per_threadgroup", "uint3"},
  };

  std::vector<std::string> attributes;
  for (const auto& [attr, dtype] : metal_attributes) {
    if (source.find(attr) != std::string::npos) {
      attributes.push_back("  " + dtype + " " + attr + " [[" + attr + "]]");
    }
  }

  return [=,
          shape_infos = std::move(shape_infos),
          attributes = std::move(attributes)](
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
      msg << "[metal_kernel] Expected `inputs` to have size "
          << input_names.size() << " but got size " << inputs.size() << "."
          << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_shapes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `output_shapes` to have size "
          << output_names.size() << " but got size " << output_shapes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_dtypes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `output_dtypes` to have size "
          << output_names.size() << " but got size " << output_dtypes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }

    auto s = resolve_metal_kernel_stream(s_);

    std::string kernel_name = "custom_kernel_" + name;
    std::string template_def = "";
    if (!template_args.empty()) {
      std::regex disallowed_chars("\\<|\\>|(, )");
      template_def = write_template(template_args);
      auto template_hash =
          std::regex_replace(template_def, disallowed_chars, "_");
      template_hash.pop_back();
      kernel_name += "_";
      kernel_name += template_hash;
    }

    // The generated source depends on the dtypes of the inputs and outputs
    // and on how each input is passed (see `write_signature`). Include them
    // in the kernel name so that a given name always maps to the same source.
    for (const auto& arr : inputs) {
      kernel_name += "_";
      kernel_name += get_type_string(arr.dtype());
      if (arr.ndim() == 0) {
        kernel_name += "s";
      } else if (arr.size() < max_constant_array_size) {
        kernel_name += "c";
      }
    }
    for (const auto& dtype : output_dtypes) {
      kernel_name += "_";
      kernel_name += get_type_string(dtype);
    }

    std::string kernel_source = write_signature(
        kernel_name,
        header,
        source,
        input_names,
        inputs,
        output_names,
        output_dtypes,
        template_args,
        attributes,
        shape_infos,
        atomic_outputs);

    if (!template_args.empty()) {
      template_def = kernel_name + template_def;
      kernel_source += "\ntemplate [[host_name(\"";
      kernel_source += kernel_name;
      kernel_source += "\")]] [[kernel]] decltype(";
      kernel_source += template_def;
      kernel_source += ") ";
      kernel_source += template_def;
      kernel_source += ";\n";
    }

    if (verbose) {
      std::cout << "Generated source code for `" << name << "`:" << std::endl
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
            0,
            static_cast<int>(compile_options.math_mode)),
        std::move(inputs));
  };
}

} // namespace mlx::core::fast
