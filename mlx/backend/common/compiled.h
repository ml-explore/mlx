// Copyright © 2023-2024 Apple Inc.
#pragma once

#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx::core {

inline bool is_static_cast(const Primitive& p) {
  return (typeid(p) == typeid(Broadcast) || typeid(p) == typeid(AsType));
}

std::string build_lib_name(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids);

std::string get_type_string(Dtype d);

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

template <typename T>
void print_complex_constant(std::ostream& os, const array& x) {
  auto old_precision = os.precision();
  T constant = x.item<T>();

  os << get_type_string(x.dtype()) << "("
     << std::setprecision(std::numeric_limits<float>::digits10 + 1)
     << constant.real() << ", " << constant.imag() << ")"
     << std::setprecision(old_precision);
}

void print_constant(std::ostream& os, const array& x);

inline bool is_scalar(const array& x) {
  return x.ndim() == 0;
}

// Check if we can use a contiguous operation given inputs and the output shape
bool compiled_check_contiguity(
    const std::vector<array>& inputs,
    const Shape& shape);

// Allocate space for the outputs possibly with input donation
void compiled_allocate_outputs(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::vector<array>& inputs_,
    const std::unordered_set<uintptr_t>& constant_ids_,
    bool contiguous);

} // namespace mlx::core
