// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

struct FusedKernelBuilder {
  std::string os;
  const std::string& kernel_name;
  const std::vector<array>& inputs;
  const std::vector<array>& outputs;
  const std::vector<array>& tape;
  const std::function<bool(size_t)>& is_constant;

  void build(const char* name, bool contiguous) {
    NodeNamer namer;

    // Function parameters.
    std::vector<std::string> params;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant(i)) {
        continue;
      }
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      params.push_back(
          fmt::format("const {}* {}", dtype_to_cuda_type(x.dtype()), xname));
      if (!is_scalar(x) && !contiguous) {
        params.push_back(fmt::format(
            "const __grid_constant__ cuda::std::array<int64_t, NDIM> {}_strides",
            xname));
      }
    }
    for (const auto& x : outputs) {
      params.push_back(fmt::format(
          "{}* {}", dtype_to_cuda_type(x.dtype()), namer.get_name(x)));
    }
    if (!contiguous) {
      params.push_back(
          "const __grid_constant__ cuda::std::array<int32_t, NDIM> shape");
    }
    params.push_back("IdxT size");

    // Build function signature.
    if (contiguous) {
      os += "template <typename IdxT = uint32_t, int work_per_thread = 1>\n";
    } else {
      os +=
          "template <int NDIM, typename IdxT = uint32_t, int work_per_thread = 1>\n";
    }
    os += fmt::format("__global__ void {}(\n", kernel_name + name);
    for (size_t i = 0; i < params.size(); ++i) {
      os += "    ";
      os += params[i];
      if (i != params.size() - 1) {
        os += ",\n";
      }
    }
    os += ") {\n";

    // Index. For non contiguous kernels we create a separate index
    // variable per variable otherwise everyone uses `index`.
    os +=
        "  IdxT index = cg::this_grid().thread_rank() * work_per_thread;\n"
        "  if (index >= size) {\n"
        "    return;\n"
        "  }\n";
    if (!contiguous) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "  IdxT " + xname + "_idx = 0;\n";
      }
      os += "  {\n";
      os += "    IdxT loc = index;\n";
      os +=
          "    #pragma unroll\n"
          "    for (int i = NDIM - 1; i >= 0; i--) {\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "      " + xname + "_idx += (loc \% shape[i]) * IdxT(" + xname +
            "_strides[i]);\n";
      }
      os +=
          "      loc /= shape[i];\n"
          "    }\n"
          "  }\n";
    }

    // Vectorized read loop
    if (contiguous) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        const std::string& xname = namer.get_name(x);
        std::string type = dtype_to_cuda_type(x.dtype());
        os += fmt::format(
            "  auto vec_{0} = load_vector<work_per_thread, {1}>({0} + index, 0, size - index, 0);\n",
            xname,
            type);
      }
    }

    // Create some space for the outputs
    for (const auto& x : outputs) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      os += fmt::format(
          "  AlignedVector<{}, work_per_thread> vec_{};\n", type, xname);
    }

    // Work loop
    if (!contiguous) {
      os +=
          "\n"
          "  for (int i = 0; i < work_per_thread && index < size; i++) {\n";
    } else {
      os +=
          "\n"
          "  #pragma unroll\n"
          "  for (int i = 0; i < work_per_thread; i++) {\n";
    }

    // Read inputs.
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      std::string value;
      if (is_constant(i)) {
        std::ostringstream ss;
        print_constant(ss, x);
        value = fmt::format("static_cast<{}>({})", type, ss.str());
      } else if (is_scalar(x)) {
        value = fmt::format("{}[0]", xname);
      } else if (contiguous) {
        value = fmt::format("vec_{}[i]", xname);
      } else {
        value = fmt::format("{}[{}_idx]", xname, xname);
      }
      os += fmt::format("    {} tmp_{} = {};\n", type, xname, value);
    }

    // Write tape.
    for (const auto& x : tape) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      std::string value;
      if (is_static_cast(x.primitive())) {
        value = fmt::format(
            "static_cast<{}>(tmp_{})", type, namer.get_name(x.inputs()[0]));
      } else {
        value = x.primitive().name();
        value += "{}(";
        for (size_t i = 0; i < x.inputs().size() - 1; ++i) {
          value += fmt::format("tmp_{}, ", namer.get_name(x.inputs()[i]));
        }
        value += fmt::format("tmp_{})", namer.get_name(x.inputs().back()));
      }
      os += fmt::format("    {} tmp_{} = {};\n", type, xname, value);
    }

    // Write output.
    for (const auto& x : outputs) {
      os += fmt::format("    vec_{0}[i] = tmp_{0};\n", namer.get_name(x));
    }

    // End of work loop
    if (!contiguous) {
      os += "\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += fmt::format("    {0}_idx += {0}_strides[NDIM - 1];\n", xname);
      }
    }
    os += "  }\n";

    // Store the output to global memory
    for (const auto& x : outputs) {
      os += fmt::format(
          "  store_vector({0} + index, 0, vec_{0}, size - index);\n",
          namer.get_name(x));
    }

    os += "}\n";
  }
};

} // namespace cu

constexpr const char* g_jit_includes = R"(
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/ternary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>

#define inf cuda::std::numeric_limits<float>::infinity()
)";

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Compiled::eval_gpu");
  auto& s = stream();

  // Determine the work per thread for the vectorized reads/writes. We take it
  // as 16 over the max itemsize for the outputs. Another heuristic could be
  // over the max itemsize of all arrays.
  int max_size = 1;
  for (const auto& x : outputs) {
    max_size = (max_size > x.itemsize()) ? max_size : x.itemsize();
  }
  int work_per_thread = 16 / max_size;

  cu::JitModule& mod = cu::get_jit_module(s.device, lib_name(), [&]() {
    // Build source code.
    cu::FusedKernelBuilder builder{
        g_jit_includes, lib_name(), inputs_, outputs_, tape_, is_constant_};
    builder.os +=
        "namespace mlx::core::cu {\n\n"
        "namespace cg = cooperative_groups;\n\n";
    builder.build("_contiguous", true);
    builder.os += "\n";
    builder.build("_strided", false);
    builder.os += "\n} // namespace mlx::core::cu\n";
    // Build kernel names.
    std::vector<std::string> kernel_names;
    kernel_names.push_back(fmt::format(
        "mlx::core::cu::{}_contiguous<uint32_t, {}>",
        lib_name(),
        work_per_thread));
    kernel_names.push_back(fmt::format(
        "mlx::core::cu::{}_contiguous<int64_t, {}>",
        lib_name(),
        work_per_thread));
    for (auto wpt : std::array<int, 2>{1, work_per_thread}) {
      for (int i = 1; i <= MAX_NDIM; ++i) {
        kernel_names.push_back(fmt::format(
            "mlx::core::cu::{}_strided<{}, uint32_t, {}>", lib_name(), i, wpt));
        kernel_names.push_back(fmt::format(
            "mlx::core::cu::{}_strided<{}, int64_t, {}>", lib_name(), i, wpt));
      }
    }

    return std::make_tuple(
        false, std::move(builder.os), std::move(kernel_names));
  });

  // Collapse contiguous dims to route to a faster kernel if possible. Also
  // handle all broadcasting.
  auto [contiguous, shape, strides_vec] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Whether to use large index.
  bool large = compiled_use_large_index(inputs, outputs, contiguous);

  cu::KernelArgs args;
  // Put inputs.
  int strides_index = 1;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    const auto& x = inputs[i];
    args.append(x);
    if (!contiguous && !is_scalar(x)) {
      args.append_ptr(strides_vec[strides_index++].data());
    }
  }

  // Put outputs.
  compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);
  for (auto& x : outputs) {
    args.append(x);
  }

  // Put shape and size.
  if (!contiguous) {
    args.append_ptr(shape.data());
  }
  if (large) {
    args.append<int64_t>(outputs[0].data_size());
  } else {
    args.append<uint32_t>(outputs[0].data_size());
  }

  // Choose work per thread
  if (!contiguous && shape.back() % work_per_thread != 0) {
    work_per_thread = 1;
  }

  // Launch kernel.
  const char* index_type = large ? "int64_t" : "uint32_t";
  std::string kernel_name = fmt::format("mlx::core::cu::{}", lib_name());
  if (contiguous) {
    kernel_name +=
        fmt::format("_contiguous<{}, {}>", index_type, work_per_thread);
  } else {
    kernel_name += fmt::format(
        "_strided<{}, {}, {}>", shape.size(), index_type, work_per_thread);
  }
  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  for (const auto& out : outputs) {
    encoder.set_output_array(out);
  }

  auto kernel = mod.get_kernel(kernel_name);
  auto [num_blocks, block_dims] =
      get_launch_args(outputs[0], large, work_per_thread);
  encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, args.args());
}

} // namespace mlx::core
