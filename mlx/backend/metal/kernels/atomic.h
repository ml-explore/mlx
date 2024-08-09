// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_atomic>
#include <metal_stdlib>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Atomic utils
///////////////////////////////////////////////////////////////////////////////

#pragma METAL internals : enable
template <typename T>
constexpr constant bool is_metal_atomic = _disjunction<
    is_same<T, int>,
    is_same<T, uint>,
    is_same<T, ulong>,
    is_same<T, float>>::value;

#pragma METAL internals : disable

template <typename T, typename = void>
struct mlx_atomic {
  atomic<uint> val;
};

template <typename T>
struct mlx_atomic<T, enable_if_t<is_metal_atomic<T>>> {
  atomic<T> val;
};

///////////////////////////////////////////////////////////////////////////////
// Native metal atomics
///////////////////////////////////////////////////////////////////////////////

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC T
mlx_atomic_load_explicit(device mlx_atomic<T>* object, size_t offset) {
  return atomic_load_explicit(&(object[offset].val), memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void
mlx_atomic_store_explicit(device mlx_atomic<T>* object, T val, size_t offset) {
  atomic_store_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_and_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_and_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_or_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_or_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_min_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_min_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_max_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_max_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_add_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_add_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_mul_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  T expected = mlx_atomic_load_explicit(object, offset);
  while (!mlx_atomic_compare_exchange_weak_explicit(
      object, &expected, val * expected, offset)) {
  }
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC bool mlx_atomic_compare_exchange_weak_explicit(
    device mlx_atomic<T>* object,
    thread T* expected,
    T val,
    size_t offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val),
      expected,
      val,
      memory_order_relaxed,
      memory_order_relaxed);
}

// Specialization for float since it does not atomic_fetch_min_explicit
template <>
METAL_FUNC void mlx_atomic_fetch_min_explicit<float>(
    device mlx_atomic<float>* object,
    float val,
    size_t offset) {
  float expected = mlx_atomic_load_explicit(object, offset);
  while (val < expected) {
    if (mlx_atomic_compare_exchange_weak_explicit(
            object, &expected, val, offset)) {
      return;
    }
  }
}

// Specialization for float since it does not atomic_fetch_max_explicit
template <>
METAL_FUNC void mlx_atomic_fetch_max_explicit<float>(
    device mlx_atomic<float>* object,
    float val,
    size_t offset) {
  float expected = mlx_atomic_load_explicit(object, offset);
  while (val > expected) {
    if (mlx_atomic_compare_exchange_weak_explicit(
            object, &expected, val, offset)) {
      return;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Custom atomics
///////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T>
constexpr constant uint packing_size = sizeof(uint) / sizeof(T);

template <typename T>
union uint_or_packed {
  T val[packing_size<T>];
  uint bits;
};

template <typename T, typename Op>
struct mlx_atomic_update_helper {
  uint operator()(uint_or_packed<T> init, T update, size_t elem_offset) {
    Op op;
    init.val[elem_offset] = op(update, init.val[elem_offset]);
    return init.bits;
  }
};

template <typename T, typename Op>
METAL_FUNC void mlx_atomic_update_and_store(
    device mlx_atomic<T>* object,
    T update,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;

  mlx_atomic_update_helper<T, Op> helper;
  uint_or_packed<T> expected;
  expected.bits =
      atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);

  while (Op::condition(update, expected.val[elem_offset]) &&
         !mlx_atomic_compare_exchange_weak_explicit(
             object,
             &(expected.bits),
             helper(expected, update, elem_offset),
             pack_offset)) {
  }
}

template <typename T>
struct __None {
  static bool condition(T a, T b) {
#pragma unused(a)
#pragma unused(b)
    return true;
  }

  T operator()(T a, T b) {
#pragma unused(b)
    return a;
  }
};

template <typename T>
struct __Add {
  static bool condition(T a, T b) {
#pragma unused(a)
#pragma unused(b)
    return true;
  }

  T operator()(T a, T b) {
    return a + b;
  }
};

template <typename T>
struct __Mul {
  static bool condition(T a, T b) {
#pragma unused(a)
    return b != 0;
  }

  T operator()(T a, T b) {
    return a * b;
  }
};

template <typename T>
struct __Max {
  static bool condition(T a, T b) {
    return a > b;
  }

  T operator()(T a, T b) {
    return max(a, b);
  }
};

template <typename T>
struct __Min {
  static bool condition(T a, T b) {
    return a < b;
  }

  T operator()(T a, T b) {
    return min(a, b);
  }
};

} // namespace

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC T
mlx_atomic_load_explicit(device mlx_atomic<T>* object, size_t offset) {
  size_t pack_offset = offset / sizeof(T);
  size_t elem_offset = offset % sizeof(T);
  uint_or_packed<T> packed_val;
  packed_val.bits =
      atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);
  return packed_val.val[elem_offset];
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void
mlx_atomic_store_explicit(device mlx_atomic<T>* object, T val, size_t offset) {
  mlx_atomic_update_and_store<T, __None<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_and_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;
  uint_or_packed<T> identity;
  identity.bits = __UINT32_MAX__;
  identity.val[elem_offset] = val;

  atomic_fetch_and_explicit(
      &(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_or_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;
  uint_or_packed<T> identity;
  identity.bits = 0;
  identity.val[elem_offset] = val;

  atomic_fetch_or_explicit(
      &(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_min_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Min<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_max_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Max<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_add_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Add<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_mul_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Mul<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC bool mlx_atomic_compare_exchange_weak_explicit(
    device mlx_atomic<T>* object,
    thread uint* expected,
    uint val,
    size_t offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val),
      expected,
      val,
      memory_order_relaxed,
      memory_order_relaxed);
}
