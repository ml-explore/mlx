// Copyright © 2024 Apple Inc.

#pragma once

// MLX_API macro for controlling symbol visibility
//
// On Windows, DLL symbols must be explicitly exported. This header provides
// the MLX_API macro which:
// - Expands to __declspec(dllexport) when building the MLX DLL (MLX_EXPORT
// defined)
// - Expands to __declspec(dllimport) when using the MLX DLL
// - Expands to nothing for static library builds or non-Windows platforms
//
// Usage:
//   MLX_API array arange(double start, double stop, ...);
//   class MLX_API SomeClass { ... };

#if defined(_WIN32) || defined(_WIN64)
#if defined(MLX_STATIC)
// Static library build - no import/export decorations needed
#define MLX_API
#elif defined(MLX_EXPORT)
#define MLX_API __declspec(dllexport)
#else
#define MLX_API __declspec(dllimport)
#endif
#else
// On non-Windows platforms, symbols are visible by default
// Could use __attribute__((visibility("default"))) if needed
#define MLX_API
#endif
