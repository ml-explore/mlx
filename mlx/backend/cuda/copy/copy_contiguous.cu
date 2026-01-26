// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_s(const In* in, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = cast_to<Out>(in[0]);
    }
  } else {
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = cast_to<Out>(in[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_v(const In* in, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = cast_to<Out>(in[i]);
    }
  } else {
    auto in_vec = load_vector<N_READS>(in, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = cast_to<Out>(in_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

} // namespace cu

// Kernel pointer accessor functions - these are the SINGLE POINT where kernel
// addresses are taken. This ensures CUDA registration happens here and the same
// addresses are used everywhere.
//
// The key insight: NVCC registers kernels when their address is taken. If we
// take the address in multiple template instantiation contexts, we get multiple
// registrations of different instantiations that don't match. By centralizing
// address-taking here, we ensure consistent kernel pointers.

// Define these as global volatile pointers to force instantiation and prevent
// optimization Note: NOT in anonymous namespace so they can be accessed by
// register_copy_contiguous_kernels

// N_READS calculation based on INPUT type size:
// sizeof(In)=1 -> N_READS=16, sizeof(In)=2 -> N_READS=8,
// sizeof(In)=4 -> N_READS=4, sizeof(In)=8 -> N_READS=2

// Macros for kernel pointer declarations
#define DECL_COPY_KERNELS(InT, OutT, in_name, out_name, N_READS)          \
  volatile void* g_copy_s_##in_name##_##out_name##_u32 =                  \
      reinterpret_cast<void*>(&cu::copy_s<InT, OutT, uint32_t, N_READS>); \
  volatile void* g_copy_v_##in_name##_##out_name##_u32 =                  \
      reinterpret_cast<void*>(&cu::copy_v<InT, OutT, uint32_t, N_READS>);

// Macro for dispatch case
#define DISPATCH_COPY(in_dtype_val, out_dtype_val, in_name, out_name)       \
  if (in_dtype.val() == in_dtype_val && out_dtype.val() == out_dtype_val) { \
    return ctype == CopyType::Scalar                                        \
        ? const_cast<void*>(g_copy_s_##in_name##_##out_name##_u32)          \
        : const_cast<void*>(g_copy_v_##in_name##_##out_name##_u32);         \
  }

// ============================================================================
// KERNEL DECLARATIONS USING DECL_COPY_KERNELS MACRO
// ============================================================================
// N_READS is based on INPUT type size:
//   sizeof(In)=1 -> N_READS=16
//   sizeof(In)=2 -> N_READS=8
//   sizeof(In)=4 -> N_READS=4
//   sizeof(In)=8 -> N_READS=2

// === Same-type copies ===
DECL_COPY_KERNELS(bool, bool, bool, bool, 16)
DECL_COPY_KERNELS(int8_t, int8_t, i8, i8, 16)
DECL_COPY_KERNELS(uint8_t, uint8_t, u8, u8, 16)
DECL_COPY_KERNELS(int16_t, int16_t, i16, i16, 8)
DECL_COPY_KERNELS(uint16_t, uint16_t, u16, u16, 8)
DECL_COPY_KERNELS(__half, __half, half, half, 8)
DECL_COPY_KERNELS(__nv_bfloat16, __nv_bfloat16, bf16, bf16, 8)
DECL_COPY_KERNELS(int32_t, int32_t, i32, i32, 4)
DECL_COPY_KERNELS(uint32_t, uint32_t, u32, u32, 4)
DECL_COPY_KERNELS(float, float, float, float, 4)
DECL_COPY_KERNELS(int64_t, int64_t, i64, i64, 2)
DECL_COPY_KERNELS(uint64_t, uint64_t, u64, u64, 2)
DECL_COPY_KERNELS(double, double, double, double, 2)
DECL_COPY_KERNELS(cu::complex64_t, cu::complex64_t, c64, c64, 2)

// === int32 <-> float ===
DECL_COPY_KERNELS(int32_t, float, i32, float, 4)
DECL_COPY_KERNELS(float, int32_t, float, i32, 4)

// === float <-> half/bfloat16 ===
DECL_COPY_KERNELS(float, __half, float, half, 4)
DECL_COPY_KERNELS(__half, float, half, float, 8)
DECL_COPY_KERNELS(float, __nv_bfloat16, float, bf16, 4)
DECL_COPY_KERNELS(__nv_bfloat16, float, bf16, float, 8)

// === bool <-> all types ===
DECL_COPY_KERNELS(bool, float, bool, float, 16)
DECL_COPY_KERNELS(float, bool, float, bool, 4)
DECL_COPY_KERNELS(bool, int32_t, bool, i32, 16)
DECL_COPY_KERNELS(int32_t, bool, i32, bool, 4)
DECL_COPY_KERNELS(bool, uint8_t, bool, u8, 16)
DECL_COPY_KERNELS(uint8_t, bool, u8, bool, 16)
DECL_COPY_KERNELS(bool, int8_t, bool, i8, 16)
DECL_COPY_KERNELS(int8_t, bool, i8, bool, 16)
DECL_COPY_KERNELS(bool, uint16_t, bool, u16, 16)
DECL_COPY_KERNELS(uint16_t, bool, u16, bool, 8)
DECL_COPY_KERNELS(bool, int16_t, bool, i16, 16)
DECL_COPY_KERNELS(int16_t, bool, i16, bool, 8)
DECL_COPY_KERNELS(bool, uint32_t, bool, u32, 16)
DECL_COPY_KERNELS(uint32_t, bool, u32, bool, 4)
DECL_COPY_KERNELS(bool, int64_t, bool, i64, 16)
DECL_COPY_KERNELS(int64_t, bool, i64, bool, 2)
DECL_COPY_KERNELS(bool, uint64_t, bool, u64, 16)
DECL_COPY_KERNELS(uint64_t, bool, u64, bool, 2)
DECL_COPY_KERNELS(bool, __half, bool, half, 16)
DECL_COPY_KERNELS(__half, bool, half, bool, 8)
DECL_COPY_KERNELS(bool, __nv_bfloat16, bool, bf16, 16)
DECL_COPY_KERNELS(__nv_bfloat16, bool, bf16, bool, 8)
// === int64 <-> float16/bfloat16 ===
DECL_COPY_KERNELS(int64_t, __half, i64, half, 2)
DECL_COPY_KERNELS(__half, int64_t, half, i64, 8)
DECL_COPY_KERNELS(int64_t, __nv_bfloat16, i64, bf16, 2)
DECL_COPY_KERNELS(__nv_bfloat16, int64_t, bf16, i64, 8)

// === uint64 <-> float16/bfloat16 ===
DECL_COPY_KERNELS(uint64_t, __half, u64, half, 2)
DECL_COPY_KERNELS(__half, uint64_t, half, u64, 8)
DECL_COPY_KERNELS(uint64_t, __nv_bfloat16, u64, bf16, 2)
DECL_COPY_KERNELS(__nv_bfloat16, uint64_t, bf16, u64, 8)

// === int32 <-> float16/bfloat16 ===
DECL_COPY_KERNELS(int32_t, __half, i32, half, 4)
DECL_COPY_KERNELS(__half, int32_t, half, i32, 8)
DECL_COPY_KERNELS(int32_t, __nv_bfloat16, i32, bf16, 4)
DECL_COPY_KERNELS(__nv_bfloat16, int32_t, bf16, i32, 8)

// === uint8/int8 <-> float ===
DECL_COPY_KERNELS(uint8_t, float, u8, float, 16)
DECL_COPY_KERNELS(int8_t, float, i8, float, 16)
DECL_COPY_KERNELS(float, uint8_t, float, u8, 4)
DECL_COPY_KERNELS(float, int8_t, float, i8, 4)

// === uint8/int8 <-> float16/bfloat16 ===
DECL_COPY_KERNELS(uint8_t, __half, u8, half, 16)
DECL_COPY_KERNELS(int8_t, __half, i8, half, 16)
DECL_COPY_KERNELS(__half, uint8_t, half, u8, 8)
DECL_COPY_KERNELS(__half, int8_t, half, i8, 8)
DECL_COPY_KERNELS(uint8_t, __nv_bfloat16, u8, bf16, 16)
DECL_COPY_KERNELS(int8_t, __nv_bfloat16, i8, bf16, 16)
DECL_COPY_KERNELS(__nv_bfloat16, uint8_t, bf16, u8, 8)
DECL_COPY_KERNELS(__nv_bfloat16, int8_t, bf16, i8, 8)

// === uint32 <-> float ===
DECL_COPY_KERNELS(uint32_t, float, u32, float, 4)
DECL_COPY_KERNELS(float, uint32_t, float, u32, 4)

// === float <-> int64/uint64 ===
DECL_COPY_KERNELS(float, uint64_t, float, u64, 4)
DECL_COPY_KERNELS(uint64_t, float, u64, float, 2)
DECL_COPY_KERNELS(float, int64_t, float, i64, 4)
DECL_COPY_KERNELS(int64_t, float, i64, float, 2)

// === float <-> complex64 ===
DECL_COPY_KERNELS(float, cu::complex64_t, float, c64, 4)
DECL_COPY_KERNELS(cu::complex64_t, float, c64, float, 2)

// === int32 <-> int64 ===
DECL_COPY_KERNELS(int32_t, int64_t, i32, i64, 4)
DECL_COPY_KERNELS(int64_t, int32_t, i64, i32, 2)

// === int32 <-> uint32 ===
DECL_COPY_KERNELS(int32_t, uint32_t, i32, u32, 4)
DECL_COPY_KERNELS(uint32_t, int32_t, u32, i32, 4)

// === double <-> various types ===
DECL_COPY_KERNELS(double, int32_t, double, i32, 2)
DECL_COPY_KERNELS(double, uint32_t, double, u32, 2)
DECL_COPY_KERNELS(double, int64_t, double, i64, 2)
DECL_COPY_KERNELS(double, uint64_t, double, u64, 2)
DECL_COPY_KERNELS(double, bool, double, bool, 2)
DECL_COPY_KERNELS(double, float, double, float, 2)
DECL_COPY_KERNELS(int32_t, double, i32, double, 4)
DECL_COPY_KERNELS(uint32_t, double, u32, double, 4)
// === uint32 <-> int64 ===
DECL_COPY_KERNELS(uint32_t, int64_t, u32, i64, 4)
DECL_COPY_KERNELS(int64_t, uint32_t, i64, u32, 2)

// === int32 <-> uint64 ===
DECL_COPY_KERNELS(int32_t, uint64_t, i32, u64, 4)
DECL_COPY_KERNELS(uint64_t, int32_t, u64, i32, 2)

// === int64 <-> uint64 ===
DECL_COPY_KERNELS(int64_t, uint64_t, i64, u64, 2)
DECL_COPY_KERNELS(uint64_t, int64_t, u64, i64, 2)

// === int32 <-> uint8/int8/uint16/int16 ===
DECL_COPY_KERNELS(int32_t, uint8_t, i32, u8, 4)
DECL_COPY_KERNELS(uint8_t, int32_t, u8, i32, 16)
DECL_COPY_KERNELS(int32_t, uint16_t, i32, u16, 4)
DECL_COPY_KERNELS(uint16_t, int32_t, u16, i32, 8)
DECL_COPY_KERNELS(int32_t, int8_t, i32, i8, 4)
DECL_COPY_KERNELS(int8_t, int32_t, i8, i32, 16)
DECL_COPY_KERNELS(int32_t, int16_t, i32, i16, 4)
DECL_COPY_KERNELS(int16_t, int32_t, i16, i32, 8)

// === uint16 <-> uint32 ===
DECL_COPY_KERNELS(uint16_t, uint32_t, u16, u32, 8)
DECL_COPY_KERNELS(uint32_t, uint16_t, u32, u16, 4)

// === uint8 <-> uint16/uint32/uint64 ===
DECL_COPY_KERNELS(uint8_t, uint16_t, u8, u16, 16)
DECL_COPY_KERNELS(uint16_t, uint8_t, u16, u8, 8)
DECL_COPY_KERNELS(uint8_t, uint32_t, u8, u32, 16)
DECL_COPY_KERNELS(uint32_t, uint8_t, u32, u8, 4)
DECL_COPY_KERNELS(uint8_t, uint64_t, u8, u64, 16)
DECL_COPY_KERNELS(uint64_t, uint8_t, u64, u8, 2)

// === int8 <-> int16/int64 ===
DECL_COPY_KERNELS(int8_t, int16_t, i8, i16, 16)
DECL_COPY_KERNELS(int16_t, int8_t, i16, i8, 8)
DECL_COPY_KERNELS(int8_t, int64_t, i8, i64, 16)
DECL_COPY_KERNELS(int64_t, int8_t, i64, i8, 2)

// === int16 <-> int64 ===
DECL_COPY_KERNELS(int16_t, int64_t, i16, i64, 8)
DECL_COPY_KERNELS(int64_t, int16_t, i64, i16, 2)

// === uint16 <-> uint64 ===
DECL_COPY_KERNELS(uint16_t, uint64_t, u16, u64, 8)
DECL_COPY_KERNELS(uint64_t, uint16_t, u64, u16, 2)

// === uint32 <-> uint64 ===
DECL_COPY_KERNELS(uint32_t, uint64_t, u32, u64, 4)
DECL_COPY_KERNELS(uint64_t, uint32_t, u64, u32, 2)

// === Cross-signed conversions ===
DECL_COPY_KERNELS(uint8_t, int8_t, u8, i8, 16)
DECL_COPY_KERNELS(int8_t, uint8_t, i8, u8, 16)
DECL_COPY_KERNELS(uint16_t, int16_t, u16, i16, 8)
DECL_COPY_KERNELS(int16_t, uint16_t, i16, u16, 8)
DECL_COPY_KERNELS(uint8_t, int16_t, u8, i16, 16)
DECL_COPY_KERNELS(int16_t, uint8_t, i16, u8, 8)
DECL_COPY_KERNELS(uint8_t, int64_t, u8, i64, 16)
DECL_COPY_KERNELS(int64_t, uint8_t, i64, u8, 2)
DECL_COPY_KERNELS(int8_t, uint16_t, i8, u16, 16)
DECL_COPY_KERNELS(uint16_t, int8_t, u16, i8, 8)
// === More cross-signed conversions ===
DECL_COPY_KERNELS(int8_t, uint32_t, i8, u32, 16)
DECL_COPY_KERNELS(uint32_t, int8_t, u32, i8, 4)
DECL_COPY_KERNELS(int8_t, uint64_t, i8, u64, 16)
DECL_COPY_KERNELS(uint64_t, int8_t, u64, i8, 2)
DECL_COPY_KERNELS(uint16_t, int64_t, u16, i64, 8)
DECL_COPY_KERNELS(int64_t, uint16_t, i64, u16, 2)
DECL_COPY_KERNELS(int16_t, uint32_t, i16, u32, 8)
DECL_COPY_KERNELS(uint32_t, int16_t, u32, i16, 4)
DECL_COPY_KERNELS(int16_t, uint64_t, i16, u64, 8)
DECL_COPY_KERNELS(uint64_t, int16_t, u64, i16, 2)

// === float <-> int16/uint16 ===
DECL_COPY_KERNELS(float, int16_t, float, i16, 4)
DECL_COPY_KERNELS(int16_t, float, i16, float, 8)
DECL_COPY_KERNELS(float, uint16_t, float, u16, 4)
DECL_COPY_KERNELS(uint16_t, float, u16, float, 8)

// === int32 <-> complex64 ===
DECL_COPY_KERNELS(int32_t, cu::complex64_t, i32, c64, 4)
DECL_COPY_KERNELS(cu::complex64_t, int32_t, c64, i32, 2)

// === uint16/int16 <-> float16/bfloat16 ===
DECL_COPY_KERNELS(uint16_t, __half, u16, half, 8)
DECL_COPY_KERNELS(uint16_t, __nv_bfloat16, u16, bf16, 8)
DECL_COPY_KERNELS(__half, uint16_t, half, u16, 8)
DECL_COPY_KERNELS(__nv_bfloat16, uint16_t, bf16, u16, 8)
DECL_COPY_KERNELS(int16_t, __half, i16, half, 8)
DECL_COPY_KERNELS(int16_t, __nv_bfloat16, i16, bf16, 8)
DECL_COPY_KERNELS(__half, int16_t, half, i16, 8)
DECL_COPY_KERNELS(__nv_bfloat16, int16_t, bf16, i16, 8)

// === uint32 <-> float16/bfloat16 ===
DECL_COPY_KERNELS(uint32_t, __half, u32, half, 4)
DECL_COPY_KERNELS(uint32_t, __nv_bfloat16, u32, bf16, 4)
DECL_COPY_KERNELS(__half, uint32_t, half, u32, 8)
DECL_COPY_KERNELS(__nv_bfloat16, uint32_t, bf16, u32, 8)

// === float16 <-> bfloat16 ===
DECL_COPY_KERNELS(__half, __nv_bfloat16, half, bf16, 8)
DECL_COPY_KERNELS(__nv_bfloat16, __half, bf16, half, 8)

// === All types <-> complex64 ===
DECL_COPY_KERNELS(bool, cu::complex64_t, bool, c64, 16)
DECL_COPY_KERNELS(cu::complex64_t, bool, c64, bool, 2)
DECL_COPY_KERNELS(uint8_t, cu::complex64_t, u8, c64, 16)
DECL_COPY_KERNELS(cu::complex64_t, uint8_t, c64, u8, 2)
DECL_COPY_KERNELS(int8_t, cu::complex64_t, i8, c64, 16)
DECL_COPY_KERNELS(cu::complex64_t, int8_t, c64, i8, 2)
DECL_COPY_KERNELS(uint16_t, cu::complex64_t, u16, c64, 8)
DECL_COPY_KERNELS(cu::complex64_t, uint16_t, c64, u16, 2)
DECL_COPY_KERNELS(int16_t, cu::complex64_t, i16, c64, 8)
DECL_COPY_KERNELS(cu::complex64_t, int16_t, c64, i16, 2)
DECL_COPY_KERNELS(uint32_t, cu::complex64_t, u32, c64, 4)
DECL_COPY_KERNELS(cu::complex64_t, uint32_t, c64, u32, 2)
DECL_COPY_KERNELS(uint64_t, cu::complex64_t, u64, c64, 2)
DECL_COPY_KERNELS(cu::complex64_t, uint64_t, c64, u64, 2)
DECL_COPY_KERNELS(int64_t, cu::complex64_t, i64, c64, 2)
DECL_COPY_KERNELS(cu::complex64_t, int64_t, c64, i64, 2)
DECL_COPY_KERNELS(__half, cu::complex64_t, half, c64, 8)
DECL_COPY_KERNELS(cu::complex64_t, __half, c64, half, 2)
DECL_COPY_KERNELS(__nv_bfloat16, cu::complex64_t, bf16, c64, 8)
DECL_COPY_KERNELS(cu::complex64_t, __nv_bfloat16, c64, bf16, 2)

// === double <-> complex64 ===
DECL_COPY_KERNELS(double, cu::complex64_t, double, c64, 2)
DECL_COPY_KERNELS(cu::complex64_t, double, c64, double, 2)

// Get kernel pointer for copy_s - uses the global volatile pointers
// NOTE: On Windows/MSVC, template type dispatch through nested lambdas fails.
// The runtime dtype-based selector in copy_contiguous() should be used instead.
// This function is kept as a fallback for non-Windows platforms.
template <typename InType, typename OutType, typename IdxT, int N_READS>
void* get_copy_s_kernel() {
  // Dispatch to the correct global pointer based on type
  // Use separate if statements with early return to satisfy MSVC's return value
  // checking
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_float_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_i32_i32_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint32_t> && std::is_same_v<OutType, uint32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_u32_u32_u32);
  }
  if constexpr (
      std::is_same_v<InType, int64_t> && std::is_same_v<OutType, int64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_s_i64_i64_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint64_t> && std::is_same_v<OutType, uint64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_s_u64_u64_u32);
  }
  if constexpr (
      std::is_same_v<InType, double> && std::is_same_v<OutType, double> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_s_double_double_u32);
  }
  if constexpr (
      std::is_same_v<InType, bool> && std::is_same_v<OutType, bool> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_s_bool_bool_u32);
  }
  if constexpr (
      std::is_same_v<InType, int8_t> && std::is_same_v<OutType, int8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_s_i8_i8_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint8_t> && std::is_same_v<OutType, uint8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_s_u8_u8_u32);
  }
  if constexpr (
      std::is_same_v<InType, int16_t> && std::is_same_v<OutType, int16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_s_i16_i16_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint16_t> && std::is_same_v<OutType, uint16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_s_u16_u16_u32);
  }
  if constexpr (
      std::is_same_v<InType, __half> && std::is_same_v<OutType, __half> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_s_half_half_u32);
  }
  if constexpr (
      std::is_same_v<InType, __nv_bfloat16> &&
      std::is_same_v<OutType, __nv_bfloat16> && N_READS == 8) {
    return const_cast<void*>(g_copy_s_bf16_bf16_u32);
  }
  if constexpr (
      std::is_same_v<InType, cu::complex64_t> &&
      std::is_same_v<OutType, cu::complex64_t> && N_READS == 2) {
    return const_cast<void*>(g_copy_s_c64_c64_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_i32_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_float_i32_u32);
  }
  // Fallback: use direct kernel address (may not be registered on Windows due
  // to MSVC template issues)
  return reinterpret_cast<void*>(&cu::copy_s<InType, OutType, IdxT, N_READS>);
}

// Get kernel pointer for copy_v - uses the global volatile pointers
template <typename InType, typename OutType, typename IdxT, int N_READS>
void* get_copy_v_kernel() {
  // Use separate if statements with early return to satisfy MSVC's return value
  // checking
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_i32_i32_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint32_t> && std::is_same_v<OutType, uint32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_u32_u32_u32);
  }
  if constexpr (
      std::is_same_v<InType, int64_t> && std::is_same_v<OutType, int64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_v_i64_i64_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint64_t> && std::is_same_v<OutType, uint64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_v_u64_u64_u32);
  }
  if constexpr (
      std::is_same_v<InType, double> && std::is_same_v<OutType, double> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_v_double_double_u32);
  }
  if constexpr (
      std::is_same_v<InType, bool> && std::is_same_v<OutType, bool> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_v_bool_bool_u32);
  }
  if constexpr (
      std::is_same_v<InType, int8_t> && std::is_same_v<OutType, int8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_v_i8_i8_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint8_t> && std::is_same_v<OutType, uint8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_v_u8_u8_u32);
  }
  if constexpr (
      std::is_same_v<InType, int16_t> && std::is_same_v<OutType, int16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_i16_i16_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint16_t> && std::is_same_v<OutType, uint16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_u16_u16_u32);
  }
  if constexpr (
      std::is_same_v<InType, __half> && std::is_same_v<OutType, __half> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_half_half_u32);
  }
  if constexpr (
      std::is_same_v<InType, __nv_bfloat16> &&
      std::is_same_v<OutType, __nv_bfloat16> && N_READS == 8) {
    return const_cast<void*>(g_copy_v_bf16_bf16_u32);
  }
  if constexpr (
      std::is_same_v<InType, cu::complex64_t> &&
      std::is_same_v<OutType, cu::complex64_t> && N_READS == 2) {
    return const_cast<void*>(g_copy_v_c64_c64_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_i32_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_i32_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, __half> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_half_u32);
  }
  if constexpr (
      std::is_same_v<InType, __half> && std::is_same_v<OutType, float> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_half_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, __nv_bfloat16> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_bf16_u32);
  }
  if constexpr (
      std::is_same_v<InType, __nv_bfloat16> && std::is_same_v<OutType, float> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_bf16_float_u32);
  }
  // Fallback: use direct kernel address (may not be registered on Windows)
  return reinterpret_cast<void*>(&cu::copy_v<InType, OutType, IdxT, N_READS>);
}

// Helper function to launch copy kernel with explicit template parameters
// This avoids MSVC issues with constexpr in nested lambda contexts
// NOTE: On Windows, use the runtime dtype-based path in copy_contiguous()
// instead.
template <typename InType, typename OutType, typename IdxT, int N_READS>
void launch_copy_contiguous_kernel(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t data_size,
    const Shape& shape,
    const Strides& strides,
    bool large) {
  auto kernel = get_copy_s_kernel<InType, OutType, IdxT, N_READS>();
  if (ctype == CopyType::Vector) {
    kernel = get_copy_v_kernel<InType, OutType, IdxT, N_READS>();
  }

  auto [num_blocks, block_dims] =
      get_launch_args(data_size, shape, strides, large, N_READS);
  IdxT size_param = static_cast<IdxT>(data_size);
  void* params[] = {(void*)&in_ptr, (void*)&out_ptr, (void*)&size_param};
  encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
}

// Dispatch on N_READS at runtime to avoid MSVC constexpr issues
template <typename InType, typename OutType, typename IdxT>
void dispatch_copy_contiguous_n_reads(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t data_size,
    const Shape& shape,
    const Strides& strides,
    bool large,
    size_t type_size) {
  // Dispatch based on runtime type_size value
  // N_READS = 16 / sizeof(InType), so:
  // sizeof=1 -> N_READS=16, sizeof=2 -> N_READS=8, sizeof=4 -> N_READS=4,
  // sizeof=8 -> N_READS=2, sizeof=16+ -> N_READS=1
  if (type_size == 1) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 16>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else if (type_size == 2) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 8>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else if (type_size == 4) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 4>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else if (type_size == 8) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 2>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 1>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  }
}

// Runtime kernel selector that uses dtype enum values directly instead of
// template type matching This avoids MSVC issues with type deduction through
// nested lambdas
void* get_copy_kernel_by_dtype(
    Dtype in_dtype,
    Dtype out_dtype,
    CopyType ctype,
    bool large) {
  // For now, handle the common same-type case (in_dtype == out_dtype)
  // These are the most common cases used by Full primitive and other operations
  if (in_dtype == out_dtype && !large) {
    if (ctype == CopyType::Scalar) {
      switch (in_dtype.val()) {
        case bool_:
          return const_cast<void*>(g_copy_s_bool_bool_u32);
        case int8:
          return const_cast<void*>(g_copy_s_i8_i8_u32);
        case uint8:
          return const_cast<void*>(g_copy_s_u8_u8_u32);
        case int16:
          return const_cast<void*>(g_copy_s_i16_i16_u32);
        case uint16:
          return const_cast<void*>(g_copy_s_u16_u16_u32);
        case float16:
          return const_cast<void*>(g_copy_s_half_half_u32);
        case bfloat16:
          return const_cast<void*>(g_copy_s_bf16_bf16_u32);
        case int32:
          return const_cast<void*>(g_copy_s_i32_i32_u32);
        case uint32:
          return const_cast<void*>(g_copy_s_u32_u32_u32);
        case float32:
          return const_cast<void*>(g_copy_s_float_float_u32);
        case int64:
          return const_cast<void*>(g_copy_s_i64_i64_u32);
        case uint64:
          return const_cast<void*>(g_copy_s_u64_u64_u32);
        case float64:
          return const_cast<void*>(g_copy_s_double_double_u32);
        case complex64:
          return const_cast<void*>(g_copy_s_c64_c64_u32);
        default:
          break;
      }
    } else { // CopyType::Vector
      switch (in_dtype.val()) {
        case bool_:
          return const_cast<void*>(g_copy_v_bool_bool_u32);
        case int8:
          return const_cast<void*>(g_copy_v_i8_i8_u32);
        case uint8:
          return const_cast<void*>(g_copy_v_u8_u8_u32);
        case int16:
          return const_cast<void*>(g_copy_v_i16_i16_u32);
        case uint16:
          return const_cast<void*>(g_copy_v_u16_u16_u32);
        case float16:
          return const_cast<void*>(g_copy_v_half_half_u32);
        case bfloat16:
          return const_cast<void*>(g_copy_v_bf16_bf16_u32);
        case int32:
          return const_cast<void*>(g_copy_v_i32_i32_u32);
        case uint32:
          return const_cast<void*>(g_copy_v_u32_u32_u32);
        case float32:
          return const_cast<void*>(g_copy_v_float_float_u32);
        case int64:
          return const_cast<void*>(g_copy_v_i64_i64_u32);
        case uint64:
          return const_cast<void*>(g_copy_v_u64_u64_u32);
        case float64:
          return const_cast<void*>(g_copy_v_double_double_u32);
        case complex64:
          return const_cast<void*>(g_copy_v_c64_c64_u32);
        default:
          break;
      }
    }
  }
  // ========================================================================
  // CROSS-TYPE DISPATCH - Using the DISPATCH_COPY macro
  // ========================================================================
  if (!large) {
    // int32 <-> float
    DISPATCH_COPY(int32, float32, i32, float)
    DISPATCH_COPY(float32, int32, float, i32)

    // bool <-> all types
    DISPATCH_COPY(bool_, float32, bool, float)
    DISPATCH_COPY(float32, bool_, float, bool)
    DISPATCH_COPY(bool_, int32, bool, i32)
    DISPATCH_COPY(int32, bool_, i32, bool)
    DISPATCH_COPY(bool_, uint8, bool, u8)
    DISPATCH_COPY(uint8, bool_, u8, bool)
    DISPATCH_COPY(bool_, int8, bool, i8)
    DISPATCH_COPY(int8, bool_, i8, bool)
    DISPATCH_COPY(bool_, uint16, bool, u16)
    DISPATCH_COPY(uint16, bool_, u16, bool)
    DISPATCH_COPY(bool_, int16, bool, i16)
    DISPATCH_COPY(int16, bool_, i16, bool)
    DISPATCH_COPY(bool_, uint32, bool, u32)
    DISPATCH_COPY(uint32, bool_, u32, bool)
    DISPATCH_COPY(bool_, int64, bool, i64)
    DISPATCH_COPY(int64, bool_, i64, bool)
    DISPATCH_COPY(bool_, uint64, bool, u64)
    DISPATCH_COPY(uint64, bool_, u64, bool)
    DISPATCH_COPY(bool_, float16, bool, half)
    DISPATCH_COPY(float16, bool_, half, bool)
    DISPATCH_COPY(bool_, bfloat16, bool, bf16)
    DISPATCH_COPY(bfloat16, bool_, bf16, bool)

    // float <-> half/bfloat16
    DISPATCH_COPY(float32, float16, float, half)
    DISPATCH_COPY(float16, float32, half, float)
    DISPATCH_COPY(float32, bfloat16, float, bf16)
    DISPATCH_COPY(bfloat16, float32, bf16, float)

    // uint8/int8 <-> float
    DISPATCH_COPY(uint8, float32, u8, float)
    DISPATCH_COPY(int8, float32, i8, float)
    DISPATCH_COPY(float32, uint8, float, u8)
    DISPATCH_COPY(float32, int8, float, i8)

    // uint8/int8 <-> float16/bfloat16
    DISPATCH_COPY(uint8, float16, u8, half)
    DISPATCH_COPY(int8, float16, i8, half)
    DISPATCH_COPY(float16, uint8, half, u8)
    DISPATCH_COPY(float16, int8, half, i8)
    DISPATCH_COPY(uint8, bfloat16, u8, bf16)
    DISPATCH_COPY(int8, bfloat16, i8, bf16)
    DISPATCH_COPY(bfloat16, uint8, bf16, u8)
    DISPATCH_COPY(bfloat16, int8, bf16, i8)

    // uint32 <-> float
    DISPATCH_COPY(uint32, float32, u32, float)
    DISPATCH_COPY(float32, uint32, float, u32)

    // float32 <-> complex64
    DISPATCH_COPY(float32, complex64, float, c64)
    DISPATCH_COPY(complex64, float32, c64, float)

    // int32 <-> int64
    DISPATCH_COPY(int32, int64, i32, i64)
    DISPATCH_COPY(int64, int32, i64, i32)

    // int32 <-> uint32
    DISPATCH_COPY(int32, uint32, i32, u32)
    DISPATCH_COPY(uint32, int32, u32, i32)

    // double <-> various types
    DISPATCH_COPY(float64, int32, double, i32)
    DISPATCH_COPY(float64, uint32, double, u32)
    DISPATCH_COPY(float64, int64, double, i64)
    DISPATCH_COPY(float64, uint64, double, u64)
    DISPATCH_COPY(float64, bool_, double, bool)
    DISPATCH_COPY(float64, float32, double, float)
    DISPATCH_COPY(int32, float64, i32, double)
    DISPATCH_COPY(uint32, float64, u32, double)

    // float32 <-> uint64/int64
    DISPATCH_COPY(float32, uint64, float, u64)
    DISPATCH_COPY(uint64, float32, u64, float)
    DISPATCH_COPY(float32, int64, float, i64)
    DISPATCH_COPY(int64, float32, i64, float)

    // uint32 <-> int64
    DISPATCH_COPY(uint32, int64, u32, i64)
    DISPATCH_COPY(int64, uint32, i64, u32)

    // int32 <-> uint64
    DISPATCH_COPY(int32, uint64, i32, u64)
    DISPATCH_COPY(uint64, int32, u64, i32)

    // int64 <-> uint64
    DISPATCH_COPY(int64, uint64, i64, u64)
    DISPATCH_COPY(uint64, int64, u64, i64)

    // int32 <-> uint8/int8/uint16/int16
    DISPATCH_COPY(int32, uint8, i32, u8)
    DISPATCH_COPY(uint8, int32, u8, i32)
    DISPATCH_COPY(int32, uint16, i32, u16)
    DISPATCH_COPY(uint16, int32, u16, i32)
    DISPATCH_COPY(int32, int8, i32, i8)
    DISPATCH_COPY(int8, int32, i8, i32)
    DISPATCH_COPY(int32, int16, i32, i16)
    DISPATCH_COPY(int16, int32, i16, i32)

    // uint16 <-> uint32
    DISPATCH_COPY(uint16, uint32, u16, u32)
    DISPATCH_COPY(uint32, uint16, u32, u16)

    // uint8 <-> uint16/uint32/uint64
    DISPATCH_COPY(uint8, uint16, u8, u16)
    DISPATCH_COPY(uint16, uint8, u16, u8)
    DISPATCH_COPY(uint8, uint32, u8, u32)
    DISPATCH_COPY(uint32, uint8, u32, u8)
    DISPATCH_COPY(uint8, uint64, u8, u64)
    DISPATCH_COPY(uint64, uint8, u64, u8)

    // int8 <-> int16/int64
    DISPATCH_COPY(int8, int16, i8, i16)
    DISPATCH_COPY(int16, int8, i16, i8)
    DISPATCH_COPY(int8, int64, i8, i64)
    DISPATCH_COPY(int64, int8, i64, i8)

    // int16 <-> int64
    DISPATCH_COPY(int16, int64, i16, i64)
    DISPATCH_COPY(int64, int16, i64, i16)

    // uint16 <-> uint64
    DISPATCH_COPY(uint16, uint64, u16, u64)
    DISPATCH_COPY(uint64, uint16, u64, u16)

    // uint32 <-> uint64
    DISPATCH_COPY(uint32, uint64, u32, u64)
    DISPATCH_COPY(uint64, uint32, u64, u32)

    // Cross-signed conversions
    DISPATCH_COPY(uint8, int8, u8, i8)
    DISPATCH_COPY(int8, uint8, i8, u8)
    DISPATCH_COPY(uint16, int16, u16, i16)
    DISPATCH_COPY(int16, uint16, i16, u16)
    DISPATCH_COPY(uint8, int16, u8, i16)
    DISPATCH_COPY(int16, uint8, i16, u8)
    DISPATCH_COPY(uint8, int64, u8, i64)
    DISPATCH_COPY(int64, uint8, i64, u8)
    DISPATCH_COPY(int8, uint16, i8, u16)
    DISPATCH_COPY(uint16, int8, u16, i8)
    DISPATCH_COPY(int8, uint32, i8, u32)
    DISPATCH_COPY(uint32, int8, u32, i8)
    DISPATCH_COPY(int8, uint64, i8, u64)
    DISPATCH_COPY(uint64, int8, u64, i8)
    DISPATCH_COPY(uint16, int64, u16, i64)
    DISPATCH_COPY(int64, uint16, i64, u16)
    DISPATCH_COPY(int16, uint32, i16, u32)
    DISPATCH_COPY(uint32, int16, u32, i16)
    DISPATCH_COPY(int16, uint64, i16, u64)
    DISPATCH_COPY(uint64, int16, u64, i16)
    DISPATCH_COPY(uint32, int64, u32, i64)

    // float32 <-> int16/uint16
    DISPATCH_COPY(float32, int16, float, i16)
    DISPATCH_COPY(int16, float32, i16, float)
    DISPATCH_COPY(float32, uint16, float, u16)
    DISPATCH_COPY(uint16, float32, u16, float)

    // int32 <-> complex64
    DISPATCH_COPY(int32, complex64, i32, c64)
    DISPATCH_COPY(complex64, int32, c64, i32)

    // int32/int64/uint64 <-> float16/bfloat16
    DISPATCH_COPY(int32, float16, i32, half)
    DISPATCH_COPY(float16, int32, half, i32)
    DISPATCH_COPY(int32, bfloat16, i32, bf16)
    DISPATCH_COPY(bfloat16, int32, bf16, i32)
    DISPATCH_COPY(int64, float16, i64, half)
    DISPATCH_COPY(float16, int64, half, i64)
    DISPATCH_COPY(int64, bfloat16, i64, bf16)
    DISPATCH_COPY(bfloat16, int64, bf16, i64)
    DISPATCH_COPY(uint64, float16, u64, half)
    DISPATCH_COPY(float16, uint64, half, u64)
    DISPATCH_COPY(uint64, bfloat16, u64, bf16)
    DISPATCH_COPY(bfloat16, uint64, bf16, u64)

    // uint16/int16 <-> float16/bfloat16
    DISPATCH_COPY(uint16, float16, u16, half)
    DISPATCH_COPY(uint16, bfloat16, u16, bf16)
    DISPATCH_COPY(float16, uint16, half, u16)
    DISPATCH_COPY(bfloat16, uint16, bf16, u16)
    DISPATCH_COPY(int16, float16, i16, half)
    DISPATCH_COPY(int16, bfloat16, i16, bf16)
    DISPATCH_COPY(float16, int16, half, i16)
    DISPATCH_COPY(bfloat16, int16, bf16, i16)

    // uint32 <-> float16/bfloat16
    DISPATCH_COPY(uint32, float16, u32, half)
    DISPATCH_COPY(uint32, bfloat16, u32, bf16)
    DISPATCH_COPY(float16, uint32, half, u32)
    DISPATCH_COPY(bfloat16, uint32, bf16, u32)

    // float16 <-> bfloat16
    DISPATCH_COPY(float16, bfloat16, half, bf16)
    DISPATCH_COPY(bfloat16, float16, bf16, half)

    // All types <-> complex64
    DISPATCH_COPY(bool_, complex64, bool, c64)
    DISPATCH_COPY(complex64, bool_, c64, bool)
    DISPATCH_COPY(uint8, complex64, u8, c64)
    DISPATCH_COPY(complex64, uint8, c64, u8)
    DISPATCH_COPY(int8, complex64, i8, c64)
    DISPATCH_COPY(complex64, int8, c64, i8)
    DISPATCH_COPY(uint16, complex64, u16, c64)
    DISPATCH_COPY(complex64, uint16, c64, u16)
    DISPATCH_COPY(int16, complex64, i16, c64)
    DISPATCH_COPY(complex64, int16, c64, i16)
    DISPATCH_COPY(uint32, complex64, u32, c64)
    DISPATCH_COPY(complex64, uint32, c64, u32)
    DISPATCH_COPY(uint64, complex64, u64, c64)
    DISPATCH_COPY(complex64, uint64, c64, u64)
    DISPATCH_COPY(int64, complex64, i64, c64)
    DISPATCH_COPY(complex64, int64, c64, i64)
    DISPATCH_COPY(float16, complex64, half, c64)
    DISPATCH_COPY(complex64, float16, c64, half)
    DISPATCH_COPY(bfloat16, complex64, bf16, c64)
    DISPATCH_COPY(complex64, bfloat16, c64, bf16)

    // double <-> complex64
    DISPATCH_COPY(float64, complex64, double, c64)
    DISPATCH_COPY(complex64, float64, c64, double)
  }
  // Return nullptr if no match - caller will use template fallback
  return nullptr;
}

void copy_contiguous(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset) {
#ifdef _MSC_VER
  // On Windows, use runtime-selected registered kernel to work around MSVC
  // template issues. MSVC doesn't properly resolve template types through
  // nested lambdas, causing the template dispatch path to use unregistered
  // kernel addresses.
  bool large = out.data_size() > UINT32_MAX;
  void* kernel =
      get_copy_kernel_by_dtype(in.dtype(), out.dtype(), ctype, large);
  if (kernel != nullptr) {
    auto [num_blocks, block_dims] = get_launch_args(
        out.data_size(),
        out.shape(),
        out.strides(),
        large,
        16 / size_of(in.dtype()));

    // Get raw GPU pointers - must use gpu_ptr, not data<void>() which may
    // copy to managed/host memory
    const void* in_ptr_val = static_cast<const char*>(gpu_ptr<void>(in)) +
        in_offset * size_of(in.dtype());
    void* out_ptr_val = static_cast<char*>(gpu_ptr<void>(out)) +
        out_offset * size_of(out.dtype());
    uint32_t size_param = static_cast<uint32_t>(out.data_size());

    // Store pointers to the pointer values for kernel params
    const void** in_ptr_ptr = &in_ptr_val;
    void** out_ptr_ptr = &out_ptr_val;
    void* params[] = {(void*)in_ptr_ptr, (void*)out_ptr_ptr, &size_param};

    encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
    return;
  }
  // Fall through to template dispatch for unsupported type combinations
#endif

  // Compute type size outside the nested lambdas to avoid MSVC constexpr issues
  size_t in_type_size = size_of(in.dtype());
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
        using InType = cuda_type_t<MLX_GET_TYPE(in_type_tag)>;
        using OutType = cuda_type_t<MLX_GET_TYPE(out_type_tag)>;
        using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
        const InType* in_ptr = gpu_ptr<InType>(in) + in_offset;
        OutType* out_ptr = gpu_ptr<OutType>(out) + out_offset;
        dispatch_copy_contiguous_n_reads<InType, OutType, IdxT>(
            encoder,
            ctype,
            in_ptr,
            out_ptr,
            out.data_size(),
            out.shape(),
            out.strides(),
            large(),
            in_type_size);
      });
    });
  });
}

} // namespace mlx::core
