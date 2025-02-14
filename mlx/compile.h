// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

enum class CompileMode { disabled, no_simplify, no_fuse, enabled };

/** Compile takes a function and returns a compiled function. */
std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    bool shapeless = false);

std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::vector<array> (*fun)(const std::vector<array>&),
    bool shapeless = false);

// Convert capture-less lambdas to function pointers.
template <
    typename F,
    typename = std::enable_if_t<
        std::is_convertible_v<F, decltype(+std::declval<F>())>>>
std::function<std::vector<array>(const std::vector<array>&)> compile(
    F&& f,
    bool shapeless = false) {
  return compile(+f, shapeless);
}

/** Globally disable compilation.
 * Setting the environment variable ``MLX_DISABLE_COMPILE`` can also
 * be used to disable compilation.
 */
void disable_compile();

/** Globally enable compilation.
 * This will override the environment variable ``MLX_DISABLE_COMPILE``.
 */
void enable_compile();

/** Set the compiler mode to the given value. */
void set_compile_mode(CompileMode mode);
} // namespace mlx::core
