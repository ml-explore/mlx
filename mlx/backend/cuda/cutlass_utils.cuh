// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/dtype.h"

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
#define CHECK_CUTLASS_ERROR(cmd) check_cutlass_error(#cmd, (cmd))

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

} // namespace mlx::core
