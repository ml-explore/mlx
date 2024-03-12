// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/compiled_preamble.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

inline void build_kernel(
    std::ostream& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids,
    bool contiguous,
    int ndim,
    bool dynamic_dims) {
  // All outputs should have the exact same shape and will be row contiguous
  auto output_shape = outputs[0].shape();
  auto output_strides = outputs[0].strides();

  // Constants are scalars that are captured by value and cannot change
  auto is_constant = [&constant_ids](const array& x) {
    return constant_ids.find(x.id()) != constant_ids.end();
  };

  NodeNamer namer;
  bool add_indices = false;
  int cnt = 0;

  // Start the kernel
  os << "[[host_name(\"" << kernel_name << "\")]]" << std::endl
     << "[[kernel]] void " << kernel_name << "(" << std::endl;

  // Add the input arguments
  for (auto& x : inputs) {
    auto& xname = namer.get_name(x);

    // Skip constants from the input list
    if (is_constant(x)) {
      continue;
    }

    // Scalars and contiguous need no strides
    if (is_scalar(x) || contiguous) {
      os << "    device const " << get_type_string(x.dtype()) << "* " << xname
         << " [[buffer(" << cnt++ << ")]]," << std::endl;
    } else {
      add_indices = true;
      os << "    device const " << get_type_string(x.dtype()) << "* " << xname
         << " [[buffer(" << cnt++ << ")]]," << std::endl
         << "    constant const size_t* " << xname << "_strides [[buffer("
         << cnt++ << ")]]," << std::endl;
    }
  }

  // Add the output arguments
  for (auto& x : outputs) {
    os << "    device " << get_type_string(x.dtype()) << "* "
       << namer.get_name(x) << " [[buffer(" << cnt++ << ")]]," << std::endl;
  }
  // Add output strides and shape to extract the indices.
  if (!contiguous) {
    os << "    constant const size_t* output_strides [[buffer(" << cnt++
       << ")]]," << std::endl
       << "    constant const int* output_shape [[buffer(" << cnt++ << ")]],"
       << std::endl;
  }
  if (dynamic_dims) {
    os << "    constant const int& ndim [[buffer(" << cnt++ << ")]],"
       << std::endl;
  }

  // The thread index in the whole grid
  os << "    uint3 pos [[thread_position_in_grid]]," << std::endl
     << "    uint3 grid [[threads_per_grid]]) {" << std::endl
     << "  uint index = pos.x + grid.x * (pos.y + grid.y * pos.z);"
     << std::endl;

  // Extract the indices per axis to individual uints if we have arrays that
  // are broadcasted or transposed
  if (add_indices) {
    if (!dynamic_dims) {
      if (ndim == 1) {
        os << "  uint index_0 = pos.x;" << std::endl;
      } else if (ndim == 2) {
        os << "  uint index_0 = pos.y;" << std::endl
           << "  uint index_1 = pos.x;" << std::endl;
      } else if (ndim == 3) {
        os << "  uint index_0 = pos.z;" << std::endl
           << "  uint index_1 = pos.y;" << std::endl
           << "  uint index_2 = pos.x;" << std::endl;
      } else {
        for (int i = 0; i < ndim - 2; i++) {
          os << "  uint index_" << i << " = (index / uint(output_strides[" << i
             << "])) % output_shape[" << i << "];" << std::endl;
        }
        os << "  uint index_" << ndim - 2 << " = pos.y;" << std::endl
           << "  uint index_" << ndim - 1 << " = pos.x;" << std::endl;
      }
    }
  }

  // Read the inputs in tmps
  for (auto& x : inputs) {
    auto& xname = namer.get_name(x);

    if (is_constant(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = ";
      print_constant(os, x);
      os << ";" << std::endl;
    } else if (is_scalar(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[0];" << std::endl;
    } else if (contiguous) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[index];" << std::endl;
    } else if (!dynamic_dims) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[";
      os << "index_0 * " << xname << "_strides[0]";
      for (int i = 1; i < ndim; i++) {
        os << " + index_" << i << " * " << xname << "_strides[" << i << "]";
      }
      os << "];" << std::endl;
    } else {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[elem_to_loc(index, output_shape, " << xname
         << "_strides, ndim)];" << std::endl;
    }
  }

  // Actually write the computation
  for (auto& x : tape) {
    os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(x)
       << " = ";
    if (is_static_cast(x.primitive())) {
      os << "static_cast<" << get_type_string(x.dtype()) << ">(tmp_"
         << namer.get_name(x.inputs()[0]) << ");" << std::endl;
    } else {
      x.primitive().print(os);
      os << "()(";
      for (int i = 0; i < x.inputs().size() - 1; i++) {
        os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
      }
      os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
    }
  }

  // Write the outputs from tmps
  for (auto& x : outputs) {
    os << "  " << namer.get_name(x) << "[index] = tmp_" << namer.get_name(x)
       << ";" << std::endl;
  }

  // Finish the kernel
  os << "}" << std::endl;

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
  // Make the name for the kernel library
  if (kernel_lib_.empty()) {
    kernel_lib_ = build_lib_name(inputs_, outputs_, tape_, constant_ids_);
  }

  // Get the kernel if someone else built it already
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto lib = d.get_library(kernel_lib_);

  // If not we have to build it ourselves
  if (lib == nullptr) {
    std::ostringstream kernel;
    kernel << metal::get_kernel_preamble() << std::endl;
    build_kernel(
        kernel,
        kernel_lib_ + "_contiguous",
        inputs_,
        outputs_,
        tape_,
        constant_ids_,
        /* contiguous = */ true,
        /* ndim = */ 0,
        /* dynamic_dims = */ false);
    for (int i = 1; i < 8; i++) {
      build_kernel(
          kernel,
          kernel_lib_ + "_strided_" + std::to_string(i),
          inputs_,
          outputs_,
          tape_,
          constant_ids_,
          /* contiguous = */ false,
          /* ndim = */ i,
          /* dynamic_dims = */ false);
    }
    build_kernel(
        kernel,
        kernel_lib_ + "_strided_dynamic",
        inputs_,
        outputs_,
        tape_,
        constant_ids_,
        /* contiguous = */ false,
        /* ndim = */ 0,
        /* dynamic_dims = */ true);

    lib = d.get_library(kernel_lib_, kernel.str());
  }

  // Figure out which kernel we are using
  auto& output_shape = outputs[0].shape();
  bool contiguous = true;
  for (auto& x : inputs) {
    if ((!x.flags().row_contiguous || x.shape() != output_shape) &&
        !is_scalar(x)) {
      contiguous = false;
      break;
    }
  }

  // Collapse contiguous dims to route to a faster kernel if possible. Also
  // handle all broadcasting.
  std::vector<std::vector<size_t>> initial_strides;
  initial_strides.push_back(outputs[0].strides());
  std::vector<int> shape;
  std::vector<std::vector<size_t>> strides;
  if (!contiguous) {
    for (int i = 0; i < inputs.size(); i++) {
      // Skip constants.
      if (constant_ids_.find(inputs_[i].id()) != constant_ids_.end()) {
        continue;
      }
      auto& x = inputs[i];

      // Skip scalar inputs.
      if (is_scalar(x)) {
        continue;
      }

      // Broadcast the inputs to the output shape.
      std::vector<size_t> xstrides;
      int j = 0;
      for (; j < output_shape.size() - x.ndim(); j++) {
        if (output_shape[j] == 1) {
          xstrides.push_back(outputs[0].strides()[j]);
        } else {
          xstrides.push_back(0);
        }
      }
      for (int i = 0; i < x.ndim(); i++, j++) {
        if (x.shape(i) == 1) {
          if (output_shape[j] == 1) {
            xstrides.push_back(outputs[0].strides()[j]);
          } else {
            xstrides.push_back(0);
          }
        } else {
          xstrides.push_back(x.strides()[i]);
        }
      }
      initial_strides.push_back(std::move(xstrides));
    }
    std::tie(shape, strides) =
        collapse_contiguous_dims(output_shape, initial_strides);
  }

  // Get the kernel from the lib
  int ndim = shape.size();
  bool dynamic = ndim >= 8;
  auto kernel_name = kernel_lib_ + (contiguous ? "_contiguous" : "_strided_");
  if (!contiguous) {
    if (dynamic) {
      kernel_name += "dynamic";
    } else {
      kernel_name += std::to_string(shape.size());
    }
  }
  auto kernel = d.get_kernel(kernel_name, lib);
  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  // Put the inputs in
  int cnt = 0;
  int stride_idx = 1; // idx 0 is the output strides
  for (int i = 0; i < inputs.size(); i++) {
    if (constant_ids_.find(inputs_[i].id()) != constant_ids_.end()) {
      continue;
    }
    auto& x = inputs[i];
    set_array_buffer(compute_encoder, x, cnt++);
    if (!contiguous && !is_scalar(x)) {
      compute_encoder->setBytes(
          strides[stride_idx].data(),
          strides[stride_idx].size() * sizeof(size_t),
          cnt++);
      stride_idx++;
    }
  }

  // Allocate space for the outputs possibly with input donation
  {
    int o = 0;
    for (int i = 0; i < inputs.size() && o < outputs.size(); ++i) {
      auto& in = inputs[i];
      // Conditions for donation
      // - Row contiguous
      // - Donatable
      // - Correct size
      // - Not a constant
      if (in.flags().row_contiguous && in.nbytes() == outputs[o].nbytes() &&
          in.is_donatable() &&
          constant_ids_.find(inputs_[i].id()) == constant_ids_.end()) {
        outputs[o].move_shared_buffer(
            in, outputs[o].strides(), in.flags(), in.data_size());
        o++;
      }
    }
    for (; o < outputs.size(); ++o) {
      outputs[o].set_data(allocator::malloc_or_wait(outputs[o].nbytes()));
    }
  }

  // Put the outputs in
  for (auto& x : outputs) {
    set_array_buffer(compute_encoder, x, cnt++);
  }

  // Put the output shape and strides in
  if (!contiguous) {
    compute_encoder->setBytes(
        strides[0].data(), strides[0].size() * sizeof(size_t), cnt++);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), cnt++);
  }

  // Put the number of dims in if it is dynamic
  if (dynamic) {
    compute_encoder->setBytes(&ndim, sizeof(int), cnt++);
  }

  // Launch the kernel
  if (contiguous) {
    size_t nthreads = outputs[0].size();
    MTL::Size grid_dims(nthreads, 1, 1);
    MTL::Size group_dims(
        std::min(nthreads, kernel->maxTotalThreadsPerThreadgroup()), 1, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  } else {
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = outputs[0].size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
}

} // namespace mlx::core
