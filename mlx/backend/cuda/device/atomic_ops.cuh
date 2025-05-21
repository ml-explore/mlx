// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/cucomplex_math.cuh"
#include "mlx/backend/cuda/device/fp16_math.cuh"

#include <cuda/atomic>

namespace mlx::core::cu {

template <typename T>
inline __device__ void atomic_add(T* out, T val) {
  cuda::atomic_ref<T, cuda::thread_scope_device> ref(*out);
  ref += val;
}

template <typename T>
inline __device__ void atomic_prod(T* out, T val) {
  cuda::atomic_ref<T, cuda::thread_scope_device> ref(*out);
  T old = ref.load();
  while (!ref.compare_exchange_strong(old, old * val)) {
  }
}

template <typename T>
inline __device__ void atomic_max(T* out, T val) {
  cuda::atomic_ref<T, cuda::thread_scope_device> ref(*out);
  ref.fetch_max(val);
}

template <typename T>
inline __device__ void atomic_min(T* out, T val) {
  cuda::atomic_ref<T, cuda::thread_scope_device> ref(*out);
  ref.fetch_min(val);
}

// Somehow cuda::atomic_ref does not provide atomic add for following types.
template <typename T>
inline __device__ void atomic_add_general(T* out, T val) {
  cuda::atomic_ref<T, cuda::thread_scope_device> ref(*out);
  T old = ref.load();
  while (!ref.compare_exchange_strong(old, old + val)) {
  }
}

inline __device__ void atomic_add(__half* out, __half val) {
  atomicAdd(out, val);
}

inline __device__ void atomic_add(cuComplex* out, cuComplex val) {
#if __CUDA_ARCH__ < 900
  atomic_add_general(out, val);
#else
  atomicAdd(out, val);
#endif
}

inline __device__ void atomic_add(__nv_bfloat16* out, __nv_bfloat16 val) {
#if __CUDA_ARCH__ < 800
#if CCCL_VERSION >= 2008000
  atomic_add_general(out, val);
#else
  bool cccl_version_too_old_for_bfloat16_atomic_add = false;
  assert(cccl_version_too_old_for_bfloat16_atomic_add);
#endif
#else
  atomicAdd(out, val);
#endif
}

} // namespace mlx::core::cu
