// Copyright © 2024 Apple Inc.

#include "python/src/utils.h"
#include "mlx/ops.h"
#include "python/src/convert.h"

array to_array(
    const ScalarOrArray& v,
    std::optional<Dtype> dtype /* = std::nullopt */) {
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
  } else if (auto pv = std::get_if<array>(&v); pv) {
    return *pv;
  } else if (auto pv = std::get_if<
                 nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>>(&v);
             pv) {
    return nd_array_to_mlx(*pv, dtype);
  } else {
    return to_array_with_accessor(std::get<nb::object>(v));
  }
}

std::pair<array, array> to_arrays(
    const ScalarOrArray& a,
    const ScalarOrArray& b) {
  // Four cases:
  // - If both a and b are arrays leave their types alone
  // - If a is an array but b is not, treat b as a weak python type
  // - If b is an array but a is not, treat a as a weak python type
  // - If neither is an array convert to arrays but leave their types alone
  auto is_mlx_array = [](const ScalarOrArray& x) {
    return std::holds_alternative<array>(x) ||
        std::holds_alternative<nb::object>(x) &&
        nb::hasattr(std::get<nb::object>(x), "__mlx_array__");
  };
  auto get_mlx_array = [](const ScalarOrArray& x) {
    if (auto px = std::get_if<array>(&x); px) {
      return *px;
    } else {
      return nb::cast<array>(std::get<nb::object>(x).attr("__mlx_array__"));
    }
  };

  if (is_mlx_array(a)) {
    auto arr_a = get_mlx_array(a);
    if (is_mlx_array(b)) {
      auto arr_b = get_mlx_array(b);
      return {arr_a, arr_b};
    }
    return {arr_a, to_array(b, arr_a.dtype())};
  } else if (is_mlx_array(b)) {
    auto arr_b = get_mlx_array(b);
    return {to_array(a, arr_b.dtype()), arr_b};
  } else {
    return {to_array(a), to_array(b)};
  }
}

array to_array_with_accessor(nb::object obj) {
  if (nb::isinstance<array>(obj)) {
    return nb::cast<array>(obj);
  } else if (nb::hasattr(obj, "__mlx_array__")) {
    return nb::cast<array>(obj.attr("__mlx_array__")());
  } else {
    std::ostringstream msg;
    msg << "Invalid type  " << nb::type_name(obj.type()).c_str()
        << " received in array initialization.";
    throw std::invalid_argument(msg.str());
  }
}
