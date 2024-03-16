// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <limits>

namespace nb = nanobind;

void init_constants(nb::module_& m) {
  m.attr("Inf") = std::numeric_limits<double>::infinity();
  m.attr("Infinity") = std::numeric_limits<double>::infinity();
  m.attr("NAN") = NAN;
  m.attr("NINF") = -std::numeric_limits<double>::infinity();
  m.attr("NZERO") = -0.0;
  m.attr("NaN") = NAN;
  m.attr("PINF") = std::numeric_limits<double>::infinity();
  m.attr("PZERO") = 0.0;
  m.attr("e") = 2.71828182845904523536028747135266249775724709369995;
  m.attr("euler_gamma") = 0.5772156649015328606065120900824024310421;
  m.attr("inf") = std::numeric_limits<double>::infinity();
  m.attr("infty") = std::numeric_limits<double>::infinity();
  m.attr("nan") = NAN;
  m.attr("newaxis") = nb::none();
  m.attr("pi") = 3.1415926535897932384626433;
}
