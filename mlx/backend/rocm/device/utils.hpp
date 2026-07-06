// Copyright © 2025 Apple Inc.

// This file must not include any host-only code, utilities that work under both
// host and device can be put here.

#pragma once

#include "mlx/backend/rocm/device/config.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace mlx::core::rocm {

///////////////////////////////////////////////////////////////////////////////
// Type traits
///////////////////////////////////////////////////////////////////////////////

// Type traits for complex types
template <typename T>
struct is_complex : std::false_type {};

template <>
struct is_complex<hipFloatComplex> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Type traits for floating point types (including half precision)
template <typename T>
inline constexpr bool is_floating_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, __half> || std::is_same_v<T, hip_bfloat16>;

// Type traits for inexact types (floating point or complex)
template <typename T>
inline constexpr bool is_inexact_v = is_floating_v<T> || is_complex_v<T>;

// Complex type alias
template <typename T>
using complex_t = hipFloatComplex;

///////////////////////////////////////////////////////////////////////////////
// Shape and Strides types
///////////////////////////////////////////////////////////////////////////////

// HIP array type (similar to cuda::std::array)
// This is usable from both host and device code
template <typename T, int N>
struct hip_array {
  T data_[N];

#ifdef __HIPCC__
  __host__ __device__ T& operator[](int i) {
    return data_[i];
  }
  __host__ __device__ const T& operator[](int i) const {
    return data_[i];
  }
  __host__ __device__ constexpr int size() const {
    return N;
  }
  __host__ __device__ T* data() {
    return data_;
  }
  __host__ __device__ const T* data() const {
    return data_;
  }
#else
  T& operator[](int i) {
    return data_[i];
  }
  const T& operator[](int i) const {
    return data_[i];
  }
  constexpr int size() const {
    return N;
  }
  T* data() {
    return data_;
  }
  const T* data() const {
    return data_;
  }
#endif
};

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
using Shape = hip_array<int32_t, MAX_NDIM>;
using Strides = hip_array<int64_t, MAX_NDIM>;

///////////////////////////////////////////////////////////////////////////////
// Vectorized load/store
///////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
  T val[N];

#ifdef __HIPCC__
  __device__ T& operator[](int i) {
    return val[i];
  }

  __device__ T operator[](int i) const {
    return val[i];
  }
#endif
};

template <int N, typename T>
inline __host__ __device__ bool is_aligned(T* x) {
  return (reinterpret_cast<uintptr_t>(x) % (N * sizeof(T))) == 0;
}

#ifdef __HIPCC__

template <int N, typename T>
inline __device__ AlignedVector<T, N> unsafe_load_vector(
    const T* ptr,
    uint32_t offset) {
  auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
  return from[offset];
}

template <int N, typename T>
inline __device__ AlignedVector<T, N> load_vector(
    const T* ptr,
    uint32_t offset) {
  if (is_aligned<N>(ptr)) {
    auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
    return from[offset];
  } else {
    AlignedVector<T, N> v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] = ptr[offset * N + i];
    }
    return v;
  }
}

template <int N, typename T, typename SizeT>
inline __device__ AlignedVector<T, N>
load_vector(const T* ptr, uint32_t offset, SizeT size, T fallback) {
  if (is_aligned<N>(ptr) && (offset + 1) * N <= size) {
    auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
    return from[offset];
  } else {
    AlignedVector<T, N> v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] = (N * offset + i) < size ? ptr[offset * N + i] : fallback;
    }
    return v;
  }
}

template <int N, typename T, typename SizeT>
inline __device__ AlignedVector<T, N> load_vector(
    const T* ptr,
    uint32_t offset,
    SizeT size,
    int64_t stride,
    T fallback) {
  if (is_aligned<N>(ptr) && stride == 1 && (offset + 1) * N <= size) {
    auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
    return from[offset];
  } else {
    AlignedVector<T, N> v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] =
          (N * offset + i) < size ? ptr[stride * (offset * N + i)] : fallback;
    }
    return v;
  }
}

template <int N, typename T>
inline __device__ void
unsafe_store_vector(T* ptr, uint32_t offset, const AlignedVector<T, N>& vec) {
  auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
  to[offset] = vec;
}

template <int N, typename T>
inline __device__ void
store_vector(T* ptr, uint32_t offset, const AlignedVector<T, N>& vec) {
  if (is_aligned<N>(ptr)) {
    auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
    to[offset] = vec;
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      ptr[offset * N + i] = vec[i];
    }
  }
}

template <int N, typename T, typename SizeT>
inline __device__ void store_vector(
    T* ptr,
    uint32_t offset,
    const AlignedVector<T, N>& vec,
    SizeT size) {
  if (is_aligned<N>(ptr) && (offset + 1) * N <= size) {
    auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
    to[offset] = vec;
  } else {
    for (int i = 0; (offset * N + i) < size && i < N; ++i) {
      ptr[offset * N + i] = vec[i];
    }
  }
}

template <int N, typename T, typename SizeT>
inline __device__ void store_vector(
    T* ptr,
    uint32_t offset,
    const AlignedVector<T, N>& vec,
    SizeT size,
    int64_t stride) {
  if (is_aligned<N>(ptr) && (offset + 1) * N <= size && stride == 1) {
    auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
    to[offset] = vec;
  } else {
    for (int i = 0; (offset * N + i) < size && i < N; ++i) {
      ptr[stride * (offset * N + i)] = vec[i];
    }
  }
}

#endif // __HIPCC__

///////////////////////////////////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////////////////////////////////

// Ceil division - available on both host and device
template <typename T>
#ifdef __HIPCC__
__host__ __device__
#endif
    T ceildiv(T a, T b) {
  return (a + b - 1) / b;
}

// ============================================================================
// Device-only code below - only compiled when using HIP compiler
// ============================================================================
#ifdef __HIPCC__

///////////////////////////////////////////////////////////////////////////////
// Numeric limits for device code
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> {
  __device__ static float infinity() {
    unsigned int i = 0x7f800000;
    return *reinterpret_cast<float*>(&i);
  }
  __device__ static float quiet_NaN() {
    unsigned int i = 0x7fc00000;
    return *reinterpret_cast<float*>(&i);
  }
  __device__ static constexpr float lowest() {
    return -3.402823466e+38f;
  }
  __device__ static constexpr float max() {
    return 3.402823466e+38f;
  }
};

template <>
struct numeric_limits<double> {
  __device__ static double infinity() {
    unsigned long long i = 0x7ff0000000000000ULL;
    return *reinterpret_cast<double*>(&i);
  }
  __device__ static double quiet_NaN() {
    unsigned long long i = 0x7ff8000000000000ULL;
    return *reinterpret_cast<double*>(&i);
  }
  __device__ static constexpr double lowest() {
    return -1.7976931348623158e+308;
  }
  __device__ static constexpr double max() {
    return 1.7976931348623158e+308;
  }
};

template <>
struct numeric_limits<__half> {
  __device__ static __half infinity() {
    return __ushort_as_half(0x7c00);
  }
  __device__ static __half quiet_NaN() {
    return __ushort_as_half(0x7e00);
  }
  __device__ static __half lowest() {
    return __ushort_as_half(0xfbff);
  }
  __device__ static __half max() {
    return __ushort_as_half(0x7bff);
  }
};

template <>
struct numeric_limits<hip_bfloat16> {
  __device__ static hip_bfloat16 infinity() {
    hip_bfloat16 val;
    val.data = 0x7f80;
    return val;
  }
  __device__ static hip_bfloat16 quiet_NaN() {
    hip_bfloat16 val;
    val.data = 0x7fc0;
    return val;
  }
  __device__ static hip_bfloat16 lowest() {
    hip_bfloat16 val;
    val.data = 0xff7f;
    return val;
  }
  __device__ static hip_bfloat16 max() {
    hip_bfloat16 val;
    val.data = 0x7f7f;
    return val;
  }
};

template <>
struct numeric_limits<int32_t> {
  __device__ static constexpr int32_t lowest() {
    return INT32_MIN;
  }
  __device__ static constexpr int32_t max() {
    return INT32_MAX;
  }
};

template <>
struct numeric_limits<int64_t> {
  __device__ static constexpr int64_t lowest() {
    return INT64_MIN;
  }
  __device__ static constexpr int64_t max() {
    return INT64_MAX;
  }
};

template <>
struct numeric_limits<uint32_t> {
  __device__ static constexpr uint32_t lowest() {
    return 0;
  }
  __device__ static constexpr uint32_t max() {
    return UINT32_MAX;
  }
};

template <>
struct numeric_limits<uint64_t> {
  __device__ static constexpr uint64_t lowest() {
    return 0;
  }
  __device__ static constexpr uint64_t max() {
    return UINT64_MAX;
  }
};

template <>
struct numeric_limits<int8_t> {
  __device__ static constexpr int8_t lowest() {
    return INT8_MIN;
  }
  __device__ static constexpr int8_t max() {
    return INT8_MAX;
  }
};

template <>
struct numeric_limits<uint8_t> {
  __device__ static constexpr uint8_t lowest() {
    return 0;
  }
  __device__ static constexpr uint8_t max() {
    return UINT8_MAX;
  }
};

template <>
struct numeric_limits<int16_t> {
  __device__ static constexpr int16_t lowest() {
    return INT16_MIN;
  }
  __device__ static constexpr int16_t max() {
    return INT16_MAX;
  }
};

template <>
struct numeric_limits<uint16_t> {
  __device__ static constexpr uint16_t lowest() {
    return 0;
  }
  __device__ static constexpr uint16_t max() {
    return UINT16_MAX;
  }
};

template <>
struct numeric_limits<bool> {
  __device__ static constexpr bool lowest() {
    return false;
  }
  __device__ static constexpr bool max() {
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Type limits utils (returns infinity for floats, max for integers)
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename = void>
struct Limits {
  __device__ static T max() {
    return numeric_limits<T>::max();
  }
  __device__ static T min() {
    return numeric_limits<T>::lowest();
  }
  __device__ static T finite_max() {
    return numeric_limits<T>::max();
  }
  __device__ static T finite_min() {
    return numeric_limits<T>::lowest();
  }
};

template <typename T>
struct Limits<
    T,
    std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>> {
  __device__ static T max() {
    return numeric_limits<T>::infinity();
  }
  __device__ static T min() {
    return -numeric_limits<T>::infinity();
  }
  __device__ static T finite_max() {
    return numeric_limits<T>::max();
  }
  __device__ static T finite_min() {
    return numeric_limits<T>::lowest();
  }
};

template <typename T>
struct Limits<
    T,
    std::enable_if_t<
        std::is_same_v<T, __half> || std::is_same_v<T, hip_bfloat16>>> {
  __device__ static T max() {
    return numeric_limits<T>::infinity();
  }
  __device__ static T min() {
    // Use float infinity for half types to avoid precision issues
    return static_cast<T>(-numeric_limits<float>::infinity());
  }
  __device__ static T finite_max() {
    return numeric_limits<T>::max();
  }
  __device__ static T finite_min() {
    return numeric_limits<T>::lowest();
  }
};

template <>
struct Limits<bool> {
  __device__ static bool max() {
    return true;
  }
  __device__ static bool min() {
    return false;
  }
  __device__ static bool finite_max() {
    return true;
  }
  __device__ static bool finite_min() {
    return false;
  }
};

template <>
struct numeric_limits<hipFloatComplex> {
  __device__ static hipFloatComplex lowest() {
    return make_hipFloatComplex(
        numeric_limits<float>::lowest(), numeric_limits<float>::lowest());
  }
  __device__ static hipFloatComplex max() {
    return make_hipFloatComplex(
        numeric_limits<float>::max(), numeric_limits<float>::max());
  }
};

template <>
struct Limits<hipFloatComplex> {
  __device__ static hipFloatComplex max() {
    return make_hipFloatComplex(Limits<float>::max(), Limits<float>::max());
  }
  __device__ static hipFloatComplex min() {
    return make_hipFloatComplex(Limits<float>::min(), Limits<float>::min());
  }
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

template <typename IdxT = int64_t>
__device__ IdxT
elem_to_loc(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Optimize when the ndim is known at compile time.
template <int NDIM, typename IdxT = int64_t>
__device__ IdxT
elem_to_loc_nd(IdxT elem, const int* shape, const int64_t* strides) {
  IdxT loc = 0;
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Two-array version
template <int NDIM, typename IdxT = int64_t>
__device__ void elem_to_loc_nd(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    IdxT& a_loc,
    IdxT& b_loc) {
  a_loc = 0;
  b_loc = 0;
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * IdxT(a_strides[i]);
    b_loc += dim_idx * IdxT(b_strides[i]);
    elem /= shape[i];
  }
}

// Three-array version
template <int NDIM, typename IdxT = int64_t>
__device__ void elem_to_loc_nd(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* c_strides,
    IdxT& a_loc,
    IdxT& b_loc,
    IdxT& c_loc) {
  a_loc = 0;
  b_loc = 0;
  c_loc = 0;
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * IdxT(a_strides[i]);
    b_loc += dim_idx * IdxT(b_strides[i]);
    c_loc += dim_idx * IdxT(c_strides[i]);
    elem /= shape[i];
  }
}

// Dynamic ndim two-array version
template <typename IdxT = int64_t>
__device__ void elem_to_loc(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndim,
    IdxT& a_loc,
    IdxT& b_loc) {
  a_loc = 0;
  b_loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * IdxT(a_strides[i]);
    b_loc += dim_idx * IdxT(b_strides[i]);
    elem /= shape[i];
  }
}

// Dynamic ndim three-array version
template <typename IdxT = int64_t>
__device__ void elem_to_loc(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* c_strides,
    int ndim,
    IdxT& a_loc,
    IdxT& b_loc,
    IdxT& c_loc) {
  a_loc = 0;
  b_loc = 0;
  c_loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * IdxT(a_strides[i]);
    b_loc += dim_idx * IdxT(b_strides[i]);
    c_loc += dim_idx * IdxT(c_strides[i]);
    elem /= shape[i];
  }
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int DIM, bool General = true, typename OffsetT = size_t>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, General, OffsetT> inner_looper;
  OffsetT offset{0};
  int index{0};

  __device__ LoopedElemToLoc(int dim) : dim(dim), inner_looper(dim - 1) {}

  __device__ void next(const int* shape, const int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  __device__ void next(int n, const int* shape, const int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  __device__ OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, true, OffsetT> {
  int dim;
  OffsetT offset{0};
  int index{0};

  __device__ LoopedElemToLoc(int dim) : dim(dim) {}

  __device__ void next(const int* shape, const int64_t* strides) {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  __device__ void next(int n, const int* shape, const int64_t* strides) {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  __device__ OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, false, OffsetT> {
  OffsetT offset{0};

  __device__ LoopedElemToLoc(int) {}

  __device__ void next(const int*, const int64_t* strides) {
    offset += OffsetT(strides[0]);
  }

  __device__ void next(int n, const int*, const int64_t* strides) {
    offset += n * OffsetT(strides[0]);
  }

  __device__ OffsetT location() {
    return offset;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Thread/block index helpers
///////////////////////////////////////////////////////////////////////////////

// Get the thread index in the block
__device__ inline int thread_index() {
  return threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
}

// Get the block index in the grid
__device__ inline int block_index() {
  return blockIdx.x + blockIdx.y * gridDim.x +
      blockIdx.z * gridDim.x * gridDim.y;
}

// Get the global thread index
__device__ inline int global_thread_index() {
  return thread_index() +
      block_index() * (blockDim.x * blockDim.y * blockDim.z);
}

#endif // __HIPCC__

} // namespace mlx::core::rocm
