// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcomplex.h>

namespace mlx::core::rocm {

// Arithmetic operations
struct Add {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a + b;
  }
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a - b;
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a * b;
  }
};

struct Divide {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a / b;
  }
};

struct Power {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return powf(a, b);
  }

  __device__ double operator()(double a, double b) {
    return pow(a, b);
  }
};

struct Remainder {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return fmodf(a, b);
  }

  __device__ double operator()(double a, double b) {
    return fmod(a, b);
  }
};

// Comparison operations
struct Equal {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a == b;
  }
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a != b;
  }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a > b;
  }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a >= b;
  }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a < b;
  }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a <= b;
  }
};

struct NaNEqual {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return (isnan(a) && isnan(b)) || (a == b);
  }
};

// Logic operations
struct LogicalAnd {
  __device__ bool operator()(bool a, bool b) {
    return a && b;
  }
};

struct LogicalOr {
  __device__ bool operator()(bool a, bool b) {
    return a || b;
  }
};

// Math operations
struct Maximum {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return fmaxf(a, b);
  }

  __device__ double operator()(double a, double b) {
    return fmax(a, b);
  }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return fminf(a, b);
  }

  __device__ double operator()(double a, double b) {
    return fmin(a, b);
  }
};

struct LogAddExp {
  template <typename T>
  __device__ T operator()(T a, T b) {
    T max_val = fmaxf(a, b);
    T min_val = fminf(a, b);
    if (isinf(max_val)) {
      return max_val;
    }
    return max_val + log1pf(expf(min_val - max_val));
  }

  __device__ double operator()(double a, double b) {
    double max_val = fmax(a, b);
    double min_val = fmin(a, b);
    if (isinf(max_val)) {
      return max_val;
    }
    return max_val + log1p(exp(min_val - max_val));
  }
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return atan2f(a, b);
  }

  __device__ double operator()(double a, double b) {
    return atan2(a, b);
  }
};

// Bitwise operations
struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a & b;
  }
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a | b;
  }
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a ^ b;
  }
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a << b;
  }
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a >> b;
  }
};

} // namespace mlx::core::rocm