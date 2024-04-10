#include <optional>

#include <nanobind/nanobind.h>
#include "mlx/dtype.h"

namespace nb = nanobind;
using namespace mlx::core;

template <typename T>
array array_from_list(T pl, std::optional<Dtype> dtype);