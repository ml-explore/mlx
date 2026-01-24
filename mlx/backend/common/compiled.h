// Copyright © 2023-2024 Apple Inc.
#pragma once

#include <functional>
#include <iomanip>

#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx::core {

inline bool is_static_cast(const Primitive& p) {
  return (typeid(p) == typeid(Broadcast) || typeid(p) == typeid(AsType));
}

std::string get_type_string(Dtype d);

template <typename T>
void print_float_constant(std::ostream& os, const array& x) {
  auto old_precision = os.precision();
  if constexpr (std::is_same_v<T, double>) {
    os << std::setprecision(std::numeric_limits<double>::digits10 + 1);
  } else {
    os << std::setprecision(std::numeric_limits<float>::digits10 + 1);
  }
  os << x.item<T>() << std::setprecision(old_precision);
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
    const std::function<bool(size_t)>& is_constant,
    bool contiguous);

// Collapse contiguous dims ignoring scalars and constants.
std::tuple<bool, Shape, std::vector<Strides>> compiled_collapse_contiguous_dims(
    const std::vector<array>& inputs,
    const array& out,
    const std::function<bool(size_t)>& is_constant);

// Return whether the kernel should use large index.
bool compiled_use_large_index(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    bool contiguous);

} // namespace mlx::core
