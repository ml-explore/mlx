// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_atomic>
#include <metal_simdgroup>

#define DEFINE_SIMD_REDUCE()                                             \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_reduce(T val) {                                                 \
    return simd_reduce_impl(val);                                        \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_reduce(T val) {                                                 \
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
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_store_explicit(out, val, offset);
  }
};

template <typename U = bool>
struct And {
  DEFINE_SIMD_REDUCE()

  bool simd_reduce_impl(bool val) {
    return simd_all(val);
  }

  static constexpr constant bool init = true;

  void atomic_update(
      device mlx_atomic<unsigned int>* out,
      bool val,
      int elem_idx,
      size_t offset = 0) {
    if (!val) {
      bool4_or_uint update;
      update.b = {true, true, true, true};
      update.b[elem_idx] = false;
      mlx_atomic_fetch_and_explicit(out, update.i, offset);
    }
  }

  void
  atomic_update(device mlx_atomic<bool>* out, bool val, size_t offset = 0) {
    if (!val) {
      mlx_atomic_store_explicit(out, val, offset);
    }
  }

  // Non atomic update
  void update(device bool* out, bool val) {
    *out &= val;
  }

  // Operator
  bool operator()(bool a, bool b) {
    return a && b;
  }
};

template <typename U = bool>
struct Or {
  DEFINE_SIMD_REDUCE()

  bool simd_reduce_impl(bool val) {
    return simd_any(val);
  }

  static constexpr constant bool init = false;

  void atomic_update(
      device mlx_atomic<unsigned int>* out,
      bool val,
      int elem_idx,
      size_t offset = 0) {
    if (val) {
      bool4_or_uint update;
      update.b = {false, false, false, false};
      update.b[elem_idx] = true;
      mlx_atomic_fetch_or_explicit(out, update.i, offset);
    }
  }

  void
  atomic_update(device mlx_atomic<bool>* out, bool val, size_t offset = 0) {
    if (val) {
      mlx_atomic_store_explicit(out, val, offset);
    }
  }

  // Non atomic update
  void update(device bool* out, bool val) {
    *out |= val;
  }

  // Operator
  bool operator()(bool a, bool b) {
    return a || b;
  }
};

template <typename U>
struct Sum {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) {
    return simd_sum(val);
  }

  static constexpr constant U init = U(0);

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_add_explicit(out, val, offset);
  }

  // Operator
  U operator()(U a, U b) {
    return a + b;
  }
};

template <typename U>
struct Prod {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) {
    return simd_product(val);
  }

  static constexpr constant U init = U(1);

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_mul_explicit(out, val, offset);
  }

  // Operator
  U operator()(U a, U b) {
    return a * b;
  }
};

template <typename U>
struct Min {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) {
    return simd_min(val);
  }

  static constexpr constant U init = Limits<U>::max;

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_min_explicit(out, val, offset);
  }

  // Operator
  U operator()(U a, U b) {
    return a < b ? a : b;
  }
};

template <typename U>
struct Max {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) {
    return simd_max(val);
  }

  static constexpr constant U init = Limits<U>::min;

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_max_explicit(out, val, offset);
  }

  // Operator
  U operator()(U a, U b) {
    return a > b ? a : b;
  }
};
