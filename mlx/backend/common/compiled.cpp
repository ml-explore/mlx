// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

void print_constant(std::ostream& os, const array& x) {
  switch (x.dtype()) {
    case float32:
      return print_float_constant<float>(os, x);
    case float16:
      return print_float_constant<float16_t>(os, x);
    case bfloat16:
      return print_float_constant<bfloat16_t>(os, x);
    case complex64:
      return print_complex_constant<complex64_t>(os, x);
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

std::string get_type_string(Dtype d) {
  switch (d) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case bfloat16:
      return "bfloat16_t";
    case complex64:
      return "complex64_t";
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
    default: {
      std::ostringstream msg;
      msg << "Unsupported compilation type " << d;
      throw std::runtime_error(msg.str());
    }
  }
}

std::string build_lib_name(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids) {
  NodeNamer namer;
  std::ostringstream os;
  std::ostringstream constant_hasher;

  // Fill the input names. This is not really necessary, I just like having A,
  // B, C, ... as the inputs.
  for (auto& x : inputs) {
    namer.get_name(x);
  }

  // The primitives describing the tape. For unary and binary primitives this
  // must be enough to describe the full computation.
  for (auto& a : tape) {
    // name and type of output
    os << namer.get_name(a) << kindof(a.dtype()) << a.itemsize();
    // computation performed
    a.primitive().print(os);
    // name of inputs to the function
    for (auto& inp : a.inputs()) {
      os << namer.get_name(inp);
    }
  }
  os << "_";

  for (auto& x : inputs) {
    if (constant_ids.find(x.id()) != constant_ids.end()) {
      os << "C";
      print_constant(constant_hasher, x);
    } else {
      os << (is_scalar(x) ? "S" : "V");
    }
  }
  os << "_";
  for (auto& x : inputs) {
    if (constant_ids.find(x.id()) != constant_ids.end()) {
      continue;
    }
    os << kindof(x.dtype()) << x.itemsize();
  }
  os << "_" << std::hash<std::string>{}(constant_hasher.str());

  return os.str();
}

} // namespace mlx::core
