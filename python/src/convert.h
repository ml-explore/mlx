// Copyright © 2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "mlx/array.h"

namespace nb = nanobind;
using namespace mlx::core;

array nd_array_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig> nd_array,
    std::optional<Dtype> dtype);
nb::ndarray<nb::numpy> mlx_to_np_array(const array& a);
