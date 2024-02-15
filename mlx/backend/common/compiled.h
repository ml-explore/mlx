// Copyright © 2023-2024 Apple Inc.
#pragma once

#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx::core {

inline bool is_static_cast(const Primitive& p) {
  return (
      typeid(p) == typeid(Broadcast) || typeid(p) == typeid(Copy) ||
      typeid(p) == typeid(StopGradient) || typeid(p) == typeid(AsType));
}

std::string build_lib_name(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids);

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

void print_constant(std::ostream& os, const array& x);

std::string get_type_string(Dtype d);

} // namespace mlx::core
