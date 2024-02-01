// Copyright © 2023-2024 Apple Inc.

#include <iostream>
#include <sstream>

#include "mlx/backend/metal/compiled_preamble.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

inline bool is_static_cast(const Primitive& p) {
  return (
      typeid(p) == typeid(Broadcast) || typeid(p) == typeid(Copy) ||
      typeid(p) == typeid(StopGradient) || typeid(p) == typeid(AsType));
}

inline auto get_type_string(Dtype d) {
  switch (d) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case bool_:
      return "bool";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int32_t";
    case int64:
      return "int64_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint32_t";
    case uint64:
      return "uint64_t";
    default:
      throw std::runtime_error("Unsupported type");
  }
}

template <typename T>
void print_float_constant(std::ostream& os, const array& x) {
  auto old_precision = os.precision();
  os << std::setprecision(std::numeric_limits<float>::digits10 + 1)
     << x.item<T>() << std::setprecision(old_precision);
}

template <typename T>
void print_int_constant(std::ostream& os, const array& x) {
  os << x.item<T>();
}

void print_constant(std::ostream& os, const array& x) {
  switch (x.dtype()) {
    case float16:
      return print_float_constant<float16_t>(os, x);
    case float32:
      return print_float_constant<float>(os, x);
    case int8:
      return print_int_constant<int8_t>(os, x);
    case int16:
      return print_int_constant<int16_t>(os, x);
    case int32:
      return print_int_constant<int32_t>(os, x);
    case int64:
      return print_int_constant<int64_t>(os, x);
    case uint8:
      return print_int_constant<uint8_t>(os, x);
    case uint16:
      return print_int_constant<uint16_t>(os, x);
    case uint32:
      return print_int_constant<uint32_t>(os, x);
    case uint64:
      return print_int_constant<uint64_t>(os, x);
    case bool_:
      os << std::boolalpha << x.item<bool>();
      return;
    default:
      throw std::runtime_error("Unsupported constant type");
  }
}

inline std::string build_kernel_name(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids) {
  std::ostringstream os;

  auto print_shape = [](std::ostream& os, const array& x) {
    for (auto s : x.shape()) {
      os << "_" << s;
    }
  };

  for (auto& a : tape) {
    a.primitive().print(os);
  }
  os << "_OD_" << outputs[0].ndim() << "_";

  for (auto& x : inputs) {
    if (constant_ids.find(x.id()) != constant_ids.end()) {
      continue;
    }
    os << ((x.size() == 1) ? "S" : "V");
  }

  return os.str();
}

inline std::string build_kernel(
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids,
    bool contiguous) {
  // All outputs should have the exact same shape and will be row contiguous
  auto output_shape = outputs[0].shape();
  auto output_strides = outputs[0].strides();

  // Constants are scalars that are captured by value and cannot change
  auto is_constant = [&constant_ids](const array& x) {
    return constant_ids.find(x.id()) != constant_ids.end();
  };

  // For scalar we shouldn't do the indexing things, just read at 0
  auto is_scalar = [](const array& x) { return x.size() == 1; };

  // No need for indexing for cases where the array is contiguous and of the
  // same shape as the output. Otherwise broadcasting or transpositions, we
  // need to do indexing.
  auto is_contiguous = [&output_shape](const array& x) {
    return x.flags().row_contiguous && x.shape() == output_shape;
  };

  std::ostringstream os;
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
  if (add_indices) {
    os << "    constant size_t* output_strides [[buffer(" << cnt++ << ")]],"
       << std::endl
       << "    constant int* output_shape [[buffer(" << cnt++ << ")]],"
       << std::endl;
  }

  // The thread index in the whole grid
  os << "    uint index [[thread_position_in_grid]]) {" << std::endl;

  // Extract the indices per axis to individual uints if we have arrays that
  // are broadcasted or transposed
  if (add_indices) {
    for (int i = 0; i < output_shape.size(); i++) {
      os << "  uint index_" << i << " = (index / output_strides[" << i
         << "]) % output_shape[" << i << "];" << std::endl;
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
    } else {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[";
      os << "index_0 * " << xname << "_strides[0]";
      for (int i = 1; i < output_shape.size(); i++) {
        os << " + index_" << i << " * " << xname << "_strides[" << i << "]";
      }
      os << "];" << std::endl;
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

  return os.str();
}

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (kernel_name_.empty()) {
    kernel_name_ = build_kernel_name(inputs_, outputs_, tape_, constant_ids_);
    kernel_source_ = metal::get_kernel_preamble();
    kernel_source_ += build_kernel(
        kernel_name_ + "_contiguous",
        inputs_,
        outputs_,
        tape_,
        constant_ids_,
        true);
    kernel_source_ += build_kernel(
        kernel_name_ + "_strided",
        inputs_,
        outputs_,
        tape_,
        constant_ids_,
        false);
    // std::cout << kernel_source_ << std::endl;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto lib = d.get_library(kernel_name_, kernel_source_);

  // Allocate space for the outputs
  for (auto& out : outputs) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }

  // Figure out which kernel we are using
  auto& output_shape = outputs[0].shape();
  auto& output_strides = outputs[0].strides();
  bool contiguous = true;
  for (auto& x : inputs) {
    if ((!x.flags().row_contiguous || x.shape() != output_shape) &&
        x.size() > 1) {
      contiguous = false;
      break;
    }
  }

  auto kernel_name = kernel_name_ + ((contiguous) ? "_contiguous" : "_strided");
  auto kernel = d.get_kernel(kernel_name, lib);
  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  // Put the inputs in
  int cnt = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (constant_ids_.find(inputs_[i].id()) != constant_ids_.end()) {
      continue;
    }
    auto& x = inputs[i];
    set_array_buffer(compute_encoder, x, cnt++);
    if (!contiguous && x.size() > 1) {
      // We need to handle broadcasting ourselves. We put 0 strides in the
      // beginning if dims are missing and 0 strides for dims with shapes of 1.
      std::vector<size_t> xstrides(output_shape.size() - x.ndim(), 0);
      for (int i = 0; i < x.ndim(); i++) {
        if (x.shape(i) == 1) {
          xstrides.push_back(0);
        } else {
          xstrides.push_back(x.strides()[i]);
        }
      }
      compute_encoder->setBytes(
          xstrides.data(), x.ndim() * sizeof(size_t), cnt++);
    }
  }

  // Put the outputs in
  for (auto& x : outputs) {
    set_array_buffer(compute_encoder, x, cnt++);
  }
  if (!contiguous) {
    compute_encoder->setBytes(
        outputs[0].strides().data(), outputs[0].ndim() * sizeof(size_t), cnt++);
    compute_encoder->setBytes(
        outputs[0].shape().data(), outputs[0].ndim() * sizeof(int), cnt++);
  }

  // Launch the kernel
  size_t nthreads = outputs[0].size();
  MTL::Size grid_dims(nthreads, 1, 1);
  MTL::Size group_dims(
      std::min(nthreads, kernel->maxTotalThreadsPerThreadgroup()), 1, 1);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace mlx::core
