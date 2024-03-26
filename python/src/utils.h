// Copyright © 2023-2024 Apple Inc.
#pragma once
#include <numeric>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/variant.h>

#include "mlx/array.h"

namespace nb = nanobind;

using namespace mlx::core;

using IntOrVec = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray = std::
    variant<nb::bool_, nb::int_, nb::float_, std::complex<float>, nb::object>;

inline std::vector<int> get_reduce_axes(const IntOrVec& v, int dims) {
  std::vector<int> axes;
  if (std::holds_alternative<std::monostate>(v)) {
    axes.resize(dims);
    std::iota(axes.begin(), axes.end(), 0);
  } else if (auto pv = std::get_if<int>(&v); pv) {
    axes.push_back(*pv);
  } else {
    axes = std::get<std::vector<int>>(v);
  }
  return axes;
}

inline array to_array_with_accessor(nb::object obj) {
  if (nb::hasattr(obj, "__mlx_array__")) {
    return nb::cast<array>(obj.attr("__mlx_array__")());
  } else if (nb::isinstance<array>(obj)) {
    return nb::cast<array>(obj);
  } else {
    std::ostringstream msg;
    msg << "Invalid type  " << nb::type_name(obj.type()).c_str()
        << " received in array initialization.";
    throw std::invalid_argument(msg.str());
  }
}

inline array to_array(
    const ScalarOrArray& v,
    std::optional<Dtype> dtype = std::nullopt) {
  if (auto pv = std::get_if<nb::bool_>(&v); pv) {
    return array(nb::cast<bool>(*pv), dtype.value_or(bool_));
  } else if (auto pv = std::get_if<nb::int_>(&v); pv) {
    auto out_t = dtype.value_or(int32);
    // bool_ is an exception and is always promoted
    return array(nb::cast<int>(*pv), (out_t == bool_) ? int32 : out_t);
  } else if (auto pv = std::get_if<nb::float_>(&v); pv) {
    auto out_t = dtype.value_or(float32);
    return array(
        nb::cast<float>(*pv), issubdtype(out_t, floating) ? out_t : float32);
  } else if (auto pv = std::get_if<std::complex<float>>(&v); pv) {
    return array(static_cast<complex64_t>(*pv), complex64);
  } else {
    return to_array_with_accessor(std::get<nb::object>(v));
  }
}

inline std::pair<array, array> to_arrays(
    const ScalarOrArray& a,
    const ScalarOrArray& b) {
  // Four cases:
  // - If both a and b are arrays leave their types alone
  // - If a is an array but b is not, treat b as a weak python type
  // - If b is an array but a is not, treat a as a weak python type
  // - If neither is an array convert to arrays but leave their types alone
  if (auto pa = std::get_if<nb::object>(&a); pa) {
    auto arr_a = to_array_with_accessor(*pa);
    if (auto pb = std::get_if<nb::object>(&b); pb) {
      auto arr_b = to_array_with_accessor(*pb);
      return {arr_a, arr_b};
    }
    return {arr_a, to_array(b, arr_a.dtype())};
  } else if (auto pb = std::get_if<nb::object>(&b); pb) {
    auto arr_b = to_array_with_accessor(*pb);
    return {to_array(a, arr_b.dtype()), arr_b};
  } else {
    return {to_array(a), to_array(b)};
  }
}
