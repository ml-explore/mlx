// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_atomic>
#include <metal_simdgroup>

#define DEFINE_SIMD_REDUCE()                                             \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_reduce(T val) thread {                                          \
    return simd_reduce_impl(val);                                        \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_reduce(T val) thread {                                          \
    for (short i = simd_size / 2; i > 0; i /= 2) {                       \
      val = operator()(val, simd_shuffle_down(val, i));                  \
    }                                                                    \
    return val;                                                          \
  }

static constant constexpr const uint8_t simd_size = 32;

union bool4_or_uint {
  bool4 b;
  unsigned int i;
};

struct None {
  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0)
      thread {
    mlx_atomic_store_explicit(out, val, offset);
  }
};

template <typename U = bool>
struct And {
  DEFINE_SIMD_REDUCE()

  bool simd_reduce_impl(bool val) thread {
    return simd_all(val);
  }

  static constexpr constant bool init = true;

  void atomic_update(
      device mlx_atomic<unsigned int>* out,
      bool val,
      int elem_idx,
      size_t offset = 0) thread {
    if (!val) {
      bool4_or_uint update;
      update.b = {true, true, true, true};
      update.b[elem_idx] = false;
      mlx_atomic_fetch_and_explicit(out, update.i, offset);
    }
  }

  void atomic_update(device mlx_atomic<bool>* out, bool val, size_t offset = 0)
      thread {
    if (!val) {
      mlx_atomic_store_explicit(out, val, offset);
    }
  }

  // Non atomic update
  void update(device bool* out, bool val) thread {
    *out &= val;
  }

  // Operator
  bool operator()(bool a, bool b) thread {
    return a && b;
  }
};

template <typename U = bool>
struct Or {
  DEFINE_SIMD_REDUCE()

  bool simd_reduce_impl(bool val) thread {
    return simd_any(val);
  }

  static constexpr constant bool init = false;

  void atomic_update(
      device mlx_atomic<unsigned int>* out,
      bool val,
      int elem_idx,
      size_t offset = 0) thread {
    if (val) {
      bool4_or_uint update;
      update.b = {false, false, false, false};
      update.b[elem_idx] = true;
      mlx_atomic_fetch_or_explicit(out, update.i, offset);
    }
  }

  void atomic_update(device mlx_atomic<bool>* out, bool val, size_t offset = 0)
      thread {
    if (val) {
      mlx_atomic_store_explicit(out, val, offset);
    }
  }

  // Non atomic update
  void update(device bool* out, bool val) thread {
    *out |= val;
  }

  // Operator
  bool operator()(bool a, bool b) thread {
    return a || b;
  }
};

template <typename U>
struct Sum {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) thread {
    return simd_sum(val);
  }

  static constexpr constant U init = U(0);

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0)
      thread {
    mlx_atomic_fetch_add_explicit(out, val, offset);
  }

  void atomic_update(
      device mlx_atomic<complex64_t>* out,
      complex64_t val,
      size_t offset = 0) thread {
    auto out_lanes = reinterpret_cast<device mlx_atomic<float>*>(out);
    mlx_atomic_fetch_add_explicit(out_lanes, val.real, 2 * offset);
    mlx_atomic_fetch_add_explicit(out_lanes, val.imag, 2 * offset + 1);
  }

  // Operator
  U operator()(U a, U b) thread {
    return a + b;
  }
};

template <typename U>
struct Prod {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) thread {
    return simd_product(val);
  }

  static constexpr constant U init = U(1);

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0)
      thread {
    mlx_atomic_fetch_mul_explicit(out, val, offset);
  }

  // Operator
  U operator()(U a, U b) thread {
    return a * b;
  }
};

template <typename U>
struct Min {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(
      T val) thread {
    return simd_min(val);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(
      T val) thread {
    if (simd_any(val != val)) {
      return static_cast<T>(NAN);
    }
    return simd_min(val);
  }

  static constexpr constant U init = Limits<U>::max;

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0)
      thread {
    mlx_atomic_fetch_min_explicit(out, val, offset);
  }

  // Operator
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    return a < b ? a : b;
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    if (metal::isnan(a) || metal::isnan(b)) {
      return static_cast<T>(NAN);
    } else {
      return a < b ? a : b;
    }
  }

  template <>
  complex64_t operator()(complex64_t a, complex64_t b) thread {
    bool real_is_nan = metal::isnan(a.real) || metal::isnan(b.real);
    bool imag_is_nan = metal::isnan(a.imag) || metal::isnan(b.imag);

    if (!real_is_nan && !imag_is_nan) {
      return a < b ? a : b;
    } else if (real_is_nan && !imag_is_nan) {
      return complex64_t(
          static_cast<float>(NAN), a.imag < b.imag ? a.imag : b.imag);
    } else if (!real_is_nan && imag_is_nan) {
      return complex64_t(
          a.real < b.real ? a.real : b.real, static_cast<float>(NAN));
    } else {
      return complex64_t(static_cast<float>(NAN), static_cast<float>(NAN));
    }
  };
};
template <typename U>
struct Max {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(
      T val) thread {
    return simd_max(val);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(
      T val) thread {
    if (simd_any(val != val)) {
      return static_cast<T>(NAN);
    }
    return simd_max(val);
  }

  static constexpr constant U init = Limits<U>::min;

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0)
      thread {
    mlx_atomic_fetch_max_explicit(out, val, offset);
  }

  // Operator
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    return a > b ? a : b;
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    if (metal::isnan(a) || metal::isnan(b)) {
      return static_cast<T>(NAN);
    } else {
      return a > b ? a : b;
    }
  }

  template <>
  complex64_t operator()(complex64_t a, complex64_t b) thread {
    bool real_is_nan = metal::isnan(a.real) || metal::isnan(b.real);
    bool imag_is_nan = metal::isnan(a.imag) || metal::isnan(b.imag);

    if (!real_is_nan && !imag_is_nan) {
      return a > b ? a : b;
    } else if (real_is_nan && !imag_is_nan) {
      return complex64_t(
          static_cast<float>(NAN), a.imag > b.imag ? a.imag : b.imag);
    } else if (!real_is_nan && imag_is_nan) {
      return complex64_t(
          a.real > b.real ? a.real : b.real, static_cast<float>(NAN));
    } else {
      return complex64_t(static_cast<float>(NAN), static_cast<float>(NAN));
    }
  }
};
