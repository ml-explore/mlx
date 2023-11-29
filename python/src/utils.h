#pragma once
#include <numeric>
#include <variant>

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/array.h"

namespace py = pybind11;

using namespace mlx::core;

using IntOrVec = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray =
    std::variant<py::bool_, py::int_, py::float_, std::complex<float>, array>;
static constexpr std::monostate none{};

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

inline array to_array(
    const ScalarOrArray& v,
    std::optional<Dtype> dtype = std::nullopt) {
  if (auto pv = std::get_if<py::bool_>(&v); pv) {
    return array(py::cast<bool>(*pv), dtype.value_or(bool_));
  } else if (auto pv = std::get_if<py::int_>(&v); pv) {
    auto out_t = dtype.value_or(int32);
    // bool_ is an exception and is always promoted
    return array(py::cast<int>(*pv), (out_t == bool_) ? int32 : out_t);
  } else if (auto pv = std::get_if<py::float_>(&v); pv) {
    auto out_t = dtype.value_or(float32);
    return array(
        py::cast<float>(*pv), is_floating_point(out_t) ? out_t : float32);
  } else if (auto pv = std::get_if<std::complex<float>>(&v); pv) {
    return array(static_cast<complex64_t>(*pv), complex64);
  } else {
    return std::get<array>(v);
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
  if (auto pa = std::get_if<array>(&a); pa) {
    if (auto pb = std::get_if<array>(&b); pb) {
      return {*pa, *pb};
    }
    return {*pa, to_array(b, pa->dtype())};
  } else if (auto pb = std::get_if<array>(&b); pb) {
    return {to_array(a, pb->dtype()), *pb};
  } else {
    return {to_array(a), to_array(b)};
  }
}
