// Copyright © 2023-2024 Apple Inc.

#pragma once

// Cpu compile enabled for unix and macos
#ifdef __unix__
#define CPU_COMPILE 1
#else
#include <TargetConditionals.h>
#define CPU_COMPILE !(TARGET_OS_IOS)
#endif

#include "mlx/array.h"

namespace mlx::core {

enum class CompileMode { disabled, no_simplify, no_fuse, enabled };

/** Compile takes a function and returns a compiled function. */
std::function<std::vector<array>(const std::vector<array>&)> compile(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    bool shapeless = false);

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
