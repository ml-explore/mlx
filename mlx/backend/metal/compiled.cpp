// Copyright © 2023-2024 Apple Inc.
#include <fmt/format.h>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

inline void build_kernel(
    std::string& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    bool contiguous,
    int ndim,
    bool dynamic_dims,
    bool use_big_index = false,
    int work_per_thread = 1) {
  NodeNamer namer;
  bool add_indices = false;
  int cnt = 0;

  // Start the kernel
  os += fmt::format(
      "[[host_name(\"{0}\")]]\n[[kernel]] void {0}(\n", kernel_name);

  // Add the input arguments
  for (size_t i = 0; i < inputs.size(); ++i) {
    // Skip constants from the input list
    if (is_constant(i)) {
      continue;
    }

    const auto& x = inputs[i];
    auto& xname = namer.get_name(x);

    // Scalars and contiguous need no strides
    if (!is_scalar(x) && !contiguous) {
      add_indices = true;
    }
    os += fmt::format(
        "    device const {0}* {1} [[buffer({2})]],\n",
        get_type_string(x.dtype()),
        xname,
        cnt++);
  }

  std::string idx_type = use_big_index ? "int64_t" : "uint";
  if (add_indices) {
    os += fmt::format(
        "    constant const int64_t* in_strides [[buffer({0})]],\n", cnt++);
  }

  // Add the output arguments
  for (auto& x : outputs) {
    os += fmt::format(
        "    device {0}* {1} [[buffer({2})]],\n",
        get_type_string(x.dtype()),
        namer.get_name(x),
        cnt++);
  }
  // Add output strides and shape to extract the indices.
  if (!contiguous) {
    os += fmt::format(
        "    constant const int* output_shape [[buffer({0})]],\n", cnt++);
  } else {
    os += fmt::format(
        "    constant const {0}& size [[buffer({1})]],\n", idx_type, cnt++);
  }
  if (dynamic_dims) {
    os += fmt::format("    constant const int& ndim [[buffer({0})]],\n", cnt++);
  }

  // The thread index in the whole grid
  os += "    uint3 pos [[thread_position_in_grid]],\n";
  os += "    uint3 grid [[threads_per_grid]]) {\n";

  os += fmt::format("  constexpr int N_ = {0};\n", work_per_thread);
  if (contiguous && use_big_index) {
    // This is only used for contiguous kernels which don't have
    // a third grid dimension
    os += "  int64_t index = N_ * (pos.x + grid.x * int64_t(pos.y));\n";
  } else if (contiguous) {
    os += "  uint index = N_ * pos.x;\n";
  } else if (work_per_thread > 1) {
    os += fmt::format(
        "  int xshape = output_shape[{0}];\n",
        dynamic_dims ? "ndim - 1" : std::to_string(ndim - 1));
    os += fmt::format(
        "  {0} index = N_ * pos.x + xshape * (pos.y + {0}(grid.y) * pos.z);\n",
        idx_type);
  } else {
    os += fmt::format(
        "  {0} index = pos.x + grid.x * (pos.y + {0}(grid.y) * pos.z);\n",
        idx_type);
  }
  if (work_per_thread > 1 && contiguous) {
    os += "  for (int i = 0; i < N_ && index < size; ++i) {\n";
  }

  // Read constant / contiguous inputs in tmps
  std::vector<array> nc_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    auto& x = inputs[i];
    auto& xname = namer.get_name(x);

    if (is_constant(i)) {
      auto type_str = get_type_string(x.dtype());
      std::ostringstream ss;
      print_constant(ss, x);
      os += fmt::format(
          "  auto tmp_{0} = static_cast<{1}>({2});\n",
          xname,
          get_type_string(x.dtype()),
          ss.str());
    } else if (is_scalar(x)) {
      os += fmt::format(
          "  {0} tmp_{1} = {1}[0];\n", get_type_string(x.dtype()), xname);
    } else if (contiguous) {
      os += fmt::format(
          "  {0} tmp_{1} = {1}[index];\n", get_type_string(x.dtype()), xname);
    } else {
      nc_inputs.push_back(x);
    }
  }

  // Initialize the indices for non-contiguous inputs
  for (int i = 0; i < nc_inputs.size(); ++i) {
    auto& xname = namer.get_name(nc_inputs[i]);
    os += fmt::format("  {0} index_{1} = ", idx_type, xname);
    if (ndim == 1) {
      int offset = i * ndim;
      os +=
          fmt::format("elem_to_loc_1<uint>(pos.x, in_strides[{0}]);\n", offset);
    } else if (ndim == 2) {
      int offset = i * ndim;
      os += fmt::format(
          "elem_to_loc_2<{0}>({{pos.x, pos.y}}, in_strides + {1});\n",
          idx_type,
          offset);
    } else if (ndim == 3) {
      int offset = i * ndim;
      os += fmt::format(
          "elem_to_loc_3<{0}>(pos, in_strides + {1});\n", idx_type, offset);
    } else if (!dynamic_dims) {
      int offset = (i + 1) * ndim;
      os += fmt::format(
          "N_ * pos.x * {0}(in_strides[{1}]) + pos.y * {0}(in_strides[{2}]);\n",
          idx_type,
          offset - 1,
          offset - 2);
    } else {
      os += fmt::format(
          "N_ * pos.x * {0}(in_strides[ndim * {1} + ndim - 1]) + pos.y * {0}(in_strides[ndim * {1} + ndim - 2]);\n",
          idx_type,
          i);
    }
  }

  if (!nc_inputs.empty() && (ndim > 3 || dynamic_dims)) {
    os += "  uint zpos = pos.z;\n";
    if (dynamic_dims) {
      os += "  for (int d = ndim - 3; d >= 0; --d) {\n";
    } else {
      os += fmt::format("  for (int d = {0}; d >= 0; --d) {{\n", ndim - 3);
    }
    os += "    uint l = zpos % output_shape[d];\n";
    for (int i = 0; i < nc_inputs.size(); ++i) {
      auto& xname = namer.get_name(nc_inputs[i]);
      os += fmt::format("    index_{0} += ", xname);
      if (dynamic_dims) {
        os +=
            fmt::format("l * {0}(in_strides[{1} * ndim + d]);\n", idx_type, i);
      } else {
        os +=
            fmt::format("l * {0}(in_strides[{1} + d]);\n", idx_type, i * ndim);
      }
    }
    os += "    zpos /= output_shape[d];\n  }\n";
  }

  // Open per-thread loop
  if (work_per_thread > 1 && !contiguous) {
    os +=
        "  for (int i = 0; i < N_ && (int(N_ * pos.x) + i) < xshape; ++i) {\n";
  }

  // Read non-contiguous inputs into tmps
  for (int i = 0; i < nc_inputs.size(); ++i) {
    auto& x = nc_inputs[i];
    auto& xname = namer.get_name(x);
    os += fmt::format(
        "  {0} tmp_{1} = {1}[index_{1}];\n", get_type_string(x.dtype()), xname);
  }

  // Actually write the computation
  for (auto& x : tape) {
    os += fmt::format(
        "  {0} tmp_{1} = ", get_type_string(x.dtype()), namer.get_name(x));
    if (is_static_cast(x.primitive())) {
      os += fmt::format(
          "static_cast<{0}>(tmp_{1});\n",
          get_type_string(x.dtype()),
          namer.get_name(x.inputs()[0]));
    } else {
      os += x.primitive().name();
      os += "()(";
      for (int i = 0; i < x.inputs().size() - 1; i++) {
        os += fmt::format("tmp_{0}, ", namer.get_name(x.inputs()[i]));
      }
      os += fmt::format("tmp_{0});\n", namer.get_name(x.inputs().back()));
    }
  }

  // Write the outputs from tmps
  for (auto& x : outputs) {
    os += fmt::format("  {0}[index] = tmp_{0};\n", namer.get_name(x));
  }
  // Increment indices and close per thread loop
  if (work_per_thread > 1) {
    for (int i = 0; i < nc_inputs.size(); ++i) {
      auto& x = nc_inputs[i];
      auto& xname = namer.get_name(x);
      if (!dynamic_dims) {
        os += fmt::format(
            "  index_{0} += in_strides[{1}];\n", xname, i * ndim + ndim - 1);
      } else {
        os += fmt::format(
            "  index_{0} += in_strides[{1} * ndim + ndim - 1];\n", xname, i);
      }
    }
    os += "  index++;\n  }\n";
  }

  // Finish the kernel
  os += "}\n";

  if (cnt > 31) {
    std::ostringstream msg;
    msg << "[compile] Too many inputs/outputs fused in the Metal Compiled "
        << "primitive which exhausted the available argument buffers for "
        << "the kernel. Please file an issue with the function that results "
        << "in this error. The name of the kernel is '" << kernel_name << "'";
    throw std::runtime_error(msg.str());
  }
}

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Get the kernel if someone else built it already
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto lib = d.get_library(kernel_lib_, [&]() {
    int work_per_thread = get_work_per_thread(outputs_[0].dtype());
    std::string kernel = metal::utils();
    concatenate(
        kernel, metal::unary_ops(), metal::binary_ops(), metal::ternary_ops());
    build_kernel(
        kernel,
        kernel_lib_ + "_contiguous",
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        /* contiguous = */ true,
        /* ndim = */ 0,
        /* dynamic_dims = */ false,
        /* use_big_index = */ false,
        /* work_per_thread = */ 1);
    if (work_per_thread > 1) {
      build_kernel(
          kernel,
          kernel_lib_ + "_contiguous_n",
          inputs_,
          outputs_,
          tape_,
          is_constant_,
          /* contiguous = */ true,
          /* ndim = */ 0,
          /* dynamic_dims = */ false,
          /* use_big_index = */ false,
          /* work_per_thread = */ work_per_thread);
    }
    build_kernel(
        kernel,
        kernel_lib_ + "_contiguous_large",
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        /* contiguous = */ true,
        /* ndim = */ 0,
        /* dynamic_dims = */ false,
        /* use_big_index = */ true,
        /* work_per_thread = */ work_per_thread);
    for (int i = 1; i < 8; i++) {
      build_kernel(
          kernel,
          kernel_lib_ + "_strided_" + std::to_string(i),
          inputs_,
          outputs_,
          tape_,
          is_constant_,
          /* contiguous = */ false,
          /* ndim = */ i,
          /* dynamic_dims = */ false,
          /* use_big_index = */ false,
          /* work_per_thread = */ i > 3 ? 2 : 1);
      if (i > 1) {
        build_kernel(
            kernel,
            kernel_lib_ + "_strided_" + std::to_string(i) + "_large",
            inputs_,
            outputs_,
            tape_,
            is_constant_,
            /* contiguous = */ false,
            /* ndim = */ i,
            /* dynamic_dims = */ false,
            /* use_big_index = */ true,
            /* work_per_thread = */ i > 3 ? 4 : 1);
      }
    }
    build_kernel(
        kernel,
        kernel_lib_ + "_strided_dynamic",
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        /* contiguous = */ false,
        /* ndim = */ 0,
        /* dynamic_dims = */ true,
        /* use_big_index = */ false,
        /* work_per_thread = */ 2);
    build_kernel(
        kernel,
        kernel_lib_ + "_strided_dynamic_large",
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        /* contiguous = */ false,
        /* ndim = */ 0,
        /* dynamic_dims = */ true,
        /* use_big_index = */ true,
        /* work_per_thread = */ 4);
    return kernel;
  });

  // Collapse contiguous dims to route to a faster kernel if possible. Also
  // handle all broadcasting.
  auto [contiguous, shape, strides] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Whether to use large index.
  bool large = compiled_use_large_index(inputs, outputs, contiguous);

  // Get the kernel from the lib
  int ndim = shape.size();
  bool dynamic = ndim >= 8;
  auto kernel_name = kernel_lib_ + (contiguous ? "_contiguous" : "_strided_");
  int work_per_thread = 1;
  if (!contiguous) {
    if (dynamic) {
      kernel_name += "dynamic";
    } else {
      kernel_name += std::to_string(shape.size());
    }
    work_per_thread = ndim > 3 ? (large ? 4 : 2) : 1;
  } else {
    work_per_thread =
        get_work_per_thread(outputs[0].dtype(), outputs[0].data_size());
    if (work_per_thread > 1 && !large) {
      kernel_name += "_n";
    }
  }
  if (large) {
    kernel_name += "_large";
  }
  auto kernel = d.get_kernel(kernel_name, lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Put the inputs in
  int cnt = 0;
  int stride_idx = 1; // idx 0 is the output strides
  Strides in_strides;
  for (int i = 0; i < inputs.size(); i++) {
    if (is_constant_(i)) {
      continue;
    }
    auto& x = inputs[i];
    compute_encoder.set_input_array(x, cnt++);
    if (!contiguous && !is_scalar(x)) {
      in_strides.insert(
          in_strides.end(),
          strides[stride_idx].begin(),
          strides[stride_idx].end());
      stride_idx++;
    }
  }
  if (!in_strides.empty()) {
    compute_encoder.set_vector_bytes(in_strides, cnt++);
  }

  compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);

  // Put the outputs in
  for (auto& x : outputs) {
    compute_encoder.set_output_array(x, cnt++);
  }

  // Put the output shape and strides in
  if (!contiguous) {
    compute_encoder.set_vector_bytes(shape, cnt++);
  } else {
    auto size = outputs[0].data_size();
    if (large) {
      compute_encoder.set_bytes<int64_t>(size, cnt++);
    } else {
      compute_encoder.set_bytes<int>(size, cnt++);
    }
  }

  // Put the number of dims in if it is dynamic
  if (dynamic) {
    compute_encoder.set_bytes(ndim, cnt++);
  }

  // Launch the kernel
  if (contiguous) {
    size_t nthreads = ceildiv(outputs[0].data_size(), work_per_thread);
    MTL::Size group_dims(
        std::min(nthreads, kernel->maxTotalThreadsPerThreadgroup()), 1, 1);
    MTL::Size grid_dims = large
        ? get_2d_grid_dims(
              outputs[0].shape(), outputs[0].strides(), work_per_thread)
        : MTL::Size(nthreads, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  } else {
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = outputs[0].size() / (dim0 * dim1);
    dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    int pow2;
    if (thread_group_size == 1024) {
      pow2 = 10;
    } else if (thread_group_size > 512) {
      pow2 = 9;
    } else {
      throw std::runtime_error("[Metal::compiled] Must use > 512 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest, pow2);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

} // namespace mlx::core
