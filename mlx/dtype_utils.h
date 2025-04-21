// Copyright © 2025 Apple Inc.
// Copyright © Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in
// https://github.com/pytorch/executorch/blob/main/LICENSE
//
// Forked from
// https://github.com/pytorch/executorch/blob/main/runtime/core/exec_aten/util/scalar_type_util.h

#pragma once

#include "mlx/dtype.h"

#include <fmt/format.h>

namespace mlx::core {

// Return string representation of dtype.
const char* dtype_to_string(Dtype arg);

// Macros that iterate across different subsets of Dtypes.
//
// For all of these macros, the final `_` parameter is the name of another macro
// that takes two parameters: the name of a C type, and the name of the
// corresponding Dtype enumerator.
//
// Note that these macros should use fully-qualified namespaces (starting with
// `::`) to ensure that they can be called safely in any arbitrary namespace.
#define MLX_FORALL_INT_TYPES(_) \
  _(uint8_t, uint8)             \
  _(uint16_t, uint16)           \
  _(uint32_t, uint32)           \
  _(uint64_t, uint64)           \
  _(int8_t, int8)               \
  _(int16_t, int16)             \
  _(int32_t, int32)             \
  _(int64_t, int64)

#define MLX_FORALL_FLOAT_TYPES(_) \
  _(float16_t, float16)           \
  _(float, float32)               \
  _(double, float64)              \
  _(bfloat16_t, bfloat16)

// Calls the provided macro on every Dtype, providing the C type and the
// Dtype name to each call.
//
// @param _ A macro that takes two parameters: the name of a C type, and the
//          name of the corresponding Dtype enumerator.
#define MLX_FORALL_DTYPES(_) \
  MLX_FORALL_INT_TYPES(_)    \
  MLX_FORALL_FLOAT_TYPES(_)  \
  _(bool, bool_)             \
  _(complex64_t, complex64)

// Maps Dtypes to C++ types.
template <Dtype::Val N>
struct DtypeToCppType;

#define SPECIALIZE_DtypeToCppType(CPP_TYPE, DTYPE) \
  template <>                                      \
  struct DtypeToCppType<Dtype::Val::DTYPE> {       \
    using type = CPP_TYPE;                         \
  };

MLX_FORALL_DTYPES(SPECIALIZE_DtypeToCppType)

#undef SPECIALIZE_DtypeToCppType

// Maps C++ types to Dtypes.
template <typename T>
struct CppTypeToDtype;

#define SPECIALIZE_CppTypeToDtype(CPP_TYPE, DTYPE) \
  template <>                                      \
  struct CppTypeToDtype<CPP_TYPE>                  \
      : std::integral_constant<Dtype::Val, Dtype::Val::DTYPE> {};

MLX_FORALL_DTYPES(SPECIALIZE_CppTypeToDtype)

#undef SPECIALIZE_CppTypeToDtype

// Helper macros for switch case macros (see below)
//
// These macros are not meant to be used directly. They provide an easy way to
// generate a switch statement that can handle subsets of Dtypes supported.

#define MLX_INTERNAL_SWITCH_CASE(enum_type, CTYPE_ALIAS, ...)         \
  case enum_type: {                                                   \
    using CTYPE_ALIAS = ::mlx::core::DtypeToCppType<enum_type>::type; \
    __VA_ARGS__;                                                      \
    break;                                                            \
  }

#define MLX_INTERNAL_SWITCH_CHECKED(TYPE, NAME, ...)                  \
  switch (TYPE) {                                                     \
    __VA_ARGS__                                                       \
    default:                                                          \
      throw std::invalid_argument(fmt::format(                        \
          "Unhandled dtype %s for %s", dtype_to_string(TYPE), NAME)); \
  }

#define MLX_INTERNAL_SWITCH_CASE_INT_TYPES(CTYPE_ALIAS, ...)     \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::uint8, CTYPE_ALIAS, __VA_ARGS__)  \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::uint16, CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::uint32, CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::uint64, CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::int8, CTYPE_ALIAS, __VA_ARGS__)   \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::int16, CTYPE_ALIAS, __VA_ARGS__)  \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::int32, CTYPE_ALIAS, __VA_ARGS__)  \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::int64, CTYPE_ALIAS, __VA_ARGS__)

#define MLX_INTERNAL_SWITCH_CASE_FLOAT_TYPES(CTYPE_ALIAS, ...)    \
  MLX_INTERNAL_SWITCH_CASE(                                       \
      ::mlx::core::Dtype::Val::float16, CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE(                                       \
      ::mlx::core::Dtype::Val::float32, CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE(                                       \
      ::mlx::core::Dtype::Val::float64, CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE(                                       \
      ::mlx::core::Dtype::Val::bfloat16, CTYPE_ALIAS, __VA_ARGS__)

#define MLX_INTERNAL_SWITCH_CASE_INT_FLOAT_TYPES(CTYPE_ALIAS, ...) \
  MLX_INTERNAL_SWITCH_CASE_INT_TYPES(CTYPE_ALIAS, __VA_ARGS__)     \
  MLX_INTERNAL_SWITCH_CASE_FLOAT_TYPES(CTYPE_ALIAS, __VA_ARGS__)

#define MLX_INTERNAL_SWITCH_CASE_REAL_TYPES(CTYPE_ALIAS, ...)        \
  MLX_INTERNAL_SWITCH_CASE_INT_FLOAT_TYPES(CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE(                                          \
      ::mlx::core::Dtype::Val::bool_, CTYPE_ALIAS, __VA_ARGS__)

#define MLX_INTERNAL_SWITCH_CASE_COMPLEX_TYPES(CTYPE_ALIAS, ...) \
  MLX_INTERNAL_SWITCH_CASE(                                      \
      ::mlx::core::Dtype::Val::complex64, CTYPE_ALIAS, __VA_ARGS__)

#define MLX_INTERNAL_SWITCH_CASE_ALL_TYPES(CTYPE_ALIAS, ...)    \
  MLX_INTERNAL_SWITCH_CASE_REAL_TYPES(CTYPE_ALIAS, __VA_ARGS__) \
  MLX_INTERNAL_SWITCH_CASE_COMPLEX_TYPES(CTYPE_ALIAS, __VA_ARGS__)

// Switch case macros
//
// These macros provide an easy way to generate switch statements that apply a
// common lambda function to subsets of Dtypes supported by MLX.
// The lambda function can type specialize to the ctype associated with the
// Dtype being handled through an alias passed as the CTYPE_ALIAS argument.
//
// Arguments:
//   - ADDITIONAL: Additional Dtype case to add
//   - TYPE: The Dtype to handle through the switch statement
//   - NAME: A name for this operation which will be used in error messages
//   - CTYPE_ALIAS: A typedef for the ctype associated with the Dtype.
//   - ...: A statement to be applied to each Dtype case
//
// An example usage is:
//
// MLX_SWITCH_ALL_TYPES(input.dtype(), CTYPE, {
//   output.data<CTYPE>[0] = input.data<CTYPE>[0];
// });
//
// Note that these can be nested as well:
//
// MLX_SWITCH_ALL_TYPES(input.dtype(), CTYPE_IN, {
//   MLX_SWITCH_ALL_TYPES(output.dtype(), CTYPE_OUT, {
//     output.data<CTYPE_OUT>[0] = input.data<CTYPE_IN>[0];
//   });
// });
//
// These macros are adapted from Dispatch.h in the ATen library. The primary
// difference is that the CTYPE_ALIAS argument is exposed to users, which is
// used to alias the ctype associated with the Dtype that is being handled.

#define MLX_SWITCH_ALL_TYPES(TYPE, CTYPE_ALIAS, ...) \
  switch (TYPE) { MLX_INTERNAL_SWITCH_CASE_ALL_TYPES(CTYPE_ALIAS, __VA_ARGS__) }

#define MLX_SWITCH_INT_TYPES_CHECKED(TYPE, NAME, CTYPE_ALIAS, ...) \
  MLX_INTERNAL_SWITCH_CHECKED(                                     \
      TYPE,                                                        \
      NAME,                                                        \
      MLX_INTERNAL_SWITCH_CASE_INT_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define MLX_SWITCH_FLOAT_TYPES_CHECKED(TYPE, NAME, CTYPE_ALIAS, ...) \
  MLX_INTERNAL_SWITCH_CHECKED(                                       \
      TYPE,                                                          \
      NAME,                                                          \
      MLX_INTERNAL_SWITCH_CASE_FLOAT_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define MLX_SWITCH_INT_FLOAT_TYPES_CHECKED(TYPE, NAME, CTYPE_ALIAS, ...) \
  MLX_INTERNAL_SWITCH_CHECKED(                                           \
      TYPE,                                                              \
      NAME,                                                              \
      MLX_INTERNAL_SWITCH_CASE_INT_FLOAT_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define MLX_SWITCH_REAL_TYPES_CHECKED(TYPE, NAME, CTYPE_ALIAS, ...) \
  MLX_INTERNAL_SWITCH_CHECKED(                                      \
      TYPE,                                                         \
      NAME,                                                         \
      MLX_INTERNAL_SWITCH_CASE_REAL_TYPES(CTYPE_ALIAS, __VA_ARGS__))

} // namespace mlx::core
