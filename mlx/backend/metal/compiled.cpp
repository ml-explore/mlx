// Copyright © 2023-2024 Apple Inc.

#include <iostream>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

bool is_noop(const Primitive& p) {
  return (
      typeid(p) == typeid(Broadcast) || typeid(p) == typeid(Copy) ||
      typeid(p) == typeid(StopGradient));
}

auto get_type_string(Dtype d) {
  if (d == float32) {
    return "float";
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

std::string build_kernel_name(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape) {
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
    os << ((x.size() == 1) ? "S" : "V");
  }

  return os.str();
}

std::string build_kernel(
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& real_inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    bool contiguous) {
  // All outputs should have the exact same shape and will be row contiguous
  auto output_shape = outputs[0].shape();
  auto output_strides = outputs[0].strides();

  // Constant means we already have the scalar value so include it in the
  // kernel as a constant.
  auto is_constant = [](const array& x) {
    return x.is_evaled() && x.size() == 1;
  };
  auto print_constant = [](std::ostream& os, const array& x) {
    const auto default_precision{os.precision()};
    switch (x.dtype()) {
      case float32:
        os << std::setprecision(std::numeric_limits<float>::digits10 + 1)
           << x.item<float>() << std::setprecision(default_precision);
        break;
      default:
        throw std::runtime_error("Not implemented");
    }
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

  // Start the kernel
  os << "[[kernel]] void " << kernel_name << "(" << std::endl;

  // Add the input arguments
  for (int i = 0; i < inputs.size(); i++) {
    auto& x = real_inputs[i];
    auto& tx = inputs[i];
    auto& xname = namer.get_name(tx);

    // // Constants need no argument
    // if (is_constant(tx)) {
    //   continue;
    // }

    // Scalars and contiguous need no strides
    if (is_scalar(x) || contiguous) { // is_contiguous(x)) {
      os << "    device const " << get_type_string(x.dtype()) << "* " << xname
         << "," << std::endl;
    } else {
      add_indices = true;
      os << "    device const " << get_type_string(x.dtype()) << "* " << xname
         << "," << std::endl
         << "    constant const size_t* " << xname << "_strides," << std::endl;
    }
  }

  // Add the output arguments
  for (auto& x : outputs) {
    os << "    device " << get_type_string(x.dtype()) << "* "
       << namer.get_name(x) << "," << std::endl;
  }
  if (add_indices) {
    os << "    constant size_t* output_strides," << std::endl
       << "    constant int* output_shape," << std::endl;
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
  for (int i = 0; i < inputs.size(); i++) {
    auto& x = real_inputs[i];
    auto& tx = inputs[i];
    auto& xname = namer.get_name(tx);

    if (false && is_constant(tx)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = ";
      print_constant(os, x);
      os << ";" << std::endl;
    } else if (is_scalar(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[0];" << std::endl;
    } else if (contiguous) { // is_contiguous(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(tx)
         << " = " << namer.get_name(tx) << "[index];" << std::endl;
    } else {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(tx)
         << " = " << namer.get_name(tx) << "[";
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
    if (is_noop(x.primitive())) {
      os << "tmp_" << namer.get_name(x.inputs()[0]) << ";" << std::endl;
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
  auto kernel_name = build_kernel_name(inputs_, outputs_, tape_);
  std::cout << build_kernel(
                   kernel_name, inputs_, inputs, outputs_, tape_, false)
            << std::endl;
  std::cout << build_kernel(kernel_name, inputs_, inputs, outputs_, tape_, true)
            << std::endl;

  // Just a fall-back to the original tape for now
  std::unordered_map<uintptr_t, array> trace_to_real;
  for (int i = 0; i < inputs.size(); ++i) {
    trace_to_real.insert({inputs_[i].id(), inputs[i]});
  }
  for (int i = 0; i < outputs.size(); ++i) {
    trace_to_real.insert({outputs_[i].id(), outputs[i]});
  }

  for (auto& a : tape_) {
    std::vector<array> p_inputs;
    for (auto& in : a.inputs()) {
      p_inputs.push_back(trace_to_real.at(in.id()));
    }
    // If a is an output get it from the map, otherwise create it
    // NB this is safe as long as no multi-output sub primitves are allowed
    // in Compiled
    std::vector<array> p_outputs;
    if (auto it = trace_to_real.find(a.id()); it != trace_to_real.end()) {
      p_outputs.push_back(it->second);
    } else {
      p_outputs.push_back(array(a.shape(), a.dtype(), a.primitive_ptr(), {}));
      trace_to_real.insert({a.id(), p_outputs[0]});
    }
    a.primitive().eval_gpu(p_inputs, p_outputs);
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);
  command_buffer->addCompletedHandler(
      [trace_to_real](MTL::CommandBuffer*) mutable {});
}

} // namespace mlx::core
