// Copyright © 2023 Apple Inc.

#pragma once

#include <variant>

#include "array.h"
#include "device.h"
#include "ops.h"
#include "stream.h"
#include "string.h"

namespace mlx::core::linalg {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

/*
Computes a vector norm.
    If axes = {}, x will be flattened before the norm is computed.
    Otherwise, the norm is computed over axes and the other dimensions are
treated as batch dimensions.
*/
array vector_norm(
    const array& a,
    const std::variant<double, std::string>& ord = 2.0,
    const std::vector<int>& axes = {},
    bool keepdims = false,
    StreamOrDevice s = {});
array vector_norm(
    const array& a,
    const std::variant<double, std::string>& ord = 2.0,
    bool keepdims = false,
    StreamOrDevice s = {});
array vector_norm(
    const array& a,
    const std::vector<int>& axes = {},
    bool keepdims = false,
    StreamOrDevice s = {});
array vector_norm(const array& a, bool keepdims = false, StreamOrDevice s = {});
} // namespace mlx::core::linalg