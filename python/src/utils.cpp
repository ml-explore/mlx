// Copyright © 2024 Apple Inc.

#include "python/src/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/convert.h"

mx::array to_array(
    const ScalarOrArray& v,
    std::optional<mx::Dtype> dtype /* = std::nullopt */) {
  if (auto pv = std::get_if<nb::bool_>(&v); pv) {
    return mx::array(nb::cast<bool>(*pv), dtype.value_or(mx::bool_));
  } else if (auto pv = std::get_if<nb::int_>(&v); pv) {
    auto val = nb::cast<long>(*pv);
    auto default_type = (val > std::numeric_limits<int>::max() ||
                         val < std::numeric_limits<int>::min())
        ? mx::int64
        : mx::int32;
    auto out_t = dtype.value_or(default_type);
    if (mx::issubdtype(out_t, mx::integer) && out_t.size() < 8) {
      auto info = mx::iinfo(out_t);
      if (val < info.min || val > static_cast<int64_t>(info.max)) {
        std::ostringstream msg;
        msg << "Converting " << val << " to " << out_t
            << " would result in overflow.";
        throw std::invalid_argument(msg.str());
      }
    }

    // bool_ is an exception and is always promoted
    return mx::array(val, (out_t == mx::bool_) ? mx::int32 : out_t);
  } else if (auto pv = std::get_if<nb::float_>(&v); pv) {
    auto out_t = dtype.value_or(mx::float32);
    return mx::array(
        nb::cast<float>(*pv),
        mx::issubdtype(out_t, mx::floating) ? out_t : mx::float32);
  } else if (auto pv = std::get_if<std::complex<float>>(&v); pv) {
    return mx::array(static_cast<mx::complex64_t>(*pv), mx::complex64);
  } else if (auto pv = std::get_if<mx::array>(&v); pv) {
    return *pv;
  } else if (auto pv = std::get_if<
                 nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>>(&v);
             pv) {
    return nd_array_to_mlx(*pv, dtype);
  } else {
    return to_array_with_accessor(std::get<ArrayLike>(v).obj);
  }
}

std::pair<mx::array, mx::array> to_arrays(
    const ScalarOrArray& a,
    const ScalarOrArray& b) {
  // Four cases:
  // - If both a and b are arrays leave their types alone
  // - If a is an array but b is not, treat b as a weak python type
  // - If b is an array but a is not, treat a as a weak python type
  // - If neither is an array convert to arrays but leave their types alone
  auto is_mlx_array = [](const ScalarOrArray& x) {
    return std::holds_alternative<mx::array>(x) ||
        std::holds_alternative<ArrayLike>(x) &&
        nb::hasattr(std::get<ArrayLike>(x).obj, "__mlx_array__");
  };
  auto get_mlx_array = [](const ScalarOrArray& x) {
    if (auto px = std::get_if<mx::array>(&x); px) {
      return *px;
    } else {
      return nb::cast<mx::array>(
          std::get<ArrayLike>(x).obj.attr("__mlx_array__"));
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

mx::array to_array_with_accessor(nb::object obj) {
  if (nb::isinstance<mx::array>(obj)) {
    return nb::cast<mx::array>(obj);
  } else if (nb::hasattr(obj, "__mlx_array__")) {
    return nb::cast<mx::array>(obj.attr("__mlx_array__")());
  } else {
    std::ostringstream msg;
    msg << "Invalid type " << nb::type_name(obj.type()).c_str()
        << " received in array initialization.";
    throw std::invalid_argument(msg.str());
  }
}
