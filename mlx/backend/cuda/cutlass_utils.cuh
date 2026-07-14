// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype.h"

#include <cute/int_tuple.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <fmt/format.h>

namespace mlx::core {

// Throw exception if the cutlass API does not succeed.
inline void check_cutlass_error(const char* name, cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        fmt::format(
            "{} failed with code: {}.",
            name,
            cutlass::cutlassGetStatusString(status)));
  }
}

// The macro version that prints the command that failed.
#define CHECK_CUTLASS_ERROR(cmd) ::mlx::core::check_cutlass_error(#cmd, (cmd))

// Maps CPU types to CUTLASS types.
template <typename T>
struct CTypeToCutlassType {
  using type = T;
};

template <>
struct CTypeToCutlassType<float16_t> {
  using type = cutlass::half_t;
};

template <>
struct CTypeToCutlassType<bfloat16_t> {
  using type = cutlass::bfloat16_t;
};

template <typename T>
using cutlass_type_t = typename CTypeToCutlassType<T>::type;

// Convert Dtype to CUTLASS C++ types.
inline const char* dtype_to_cutlass_type(const Dtype& dtype) {
  if (dtype == float16) {
    return "cutlass::half_t";
  }
  if (dtype == bfloat16) {
    return "cutlass::bfloat16_t";
  }
  return dtype_to_cuda_type(dtype);
}

// Convert cute shape to string.
inline auto cta_tiler_to_string(auto cta_tiler) {
  return fmt::format(
      "cute::Shape<cute::Int<{}>, cute::Int<{}>, cute::Int<{}>>",
      int(cute::size<0>(cta_tiler)),
      int(cute::size<1>(cta_tiler)),
      int(cute::size<2>(cta_tiler)));
}

} // namespace mlx::core
