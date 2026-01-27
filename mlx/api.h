// Copyright © 2024 Apple Inc.

#pragma once

// MLX_API macro for controlling symbol visibility, must add for public APIs.
//
// Usage:
//   MLX_API void some_function(...);
//   class MLX_API SomeClass { ... };

#if defined(MLX_STATIC)

// Static library build - no import/export decorations needed
#define MLX_API

#else

// Shared library build.
#if defined(_WIN32)
#if defined(MLX_EXPORT)
#define MLX_API __declspec(dllexport)
#else
#define MLX_API __declspec(dllimport)
#endif // defined(MLX_EXPORT)
#else
#define MLX_API __attribute__((visibility("default")))
#endif // defined(_WIN32)

#endif // defined(MLX_STATIC)
