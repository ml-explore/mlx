// Copyright © 2025 Apple Inc.
#pragma once

namespace mlx::core {

namespace cu {
// See:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
// cvt.rn/rs.satfinite.e2m1x2/e2m1x4/e4m3x2.f32 instructions are supported on
// the following architectures: sm_100a, sm_101a, sm_120a The macro
// __CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__ is defined in
// <cuda_fp8.hpp> and used in <cuda_fp4.hpp> / <cuda_fp8.hpp> to check whether
// these cvt instructions are supported for FP4/FP8. If the macro is not set,
// those headers fall back to emulation (manual double -> FP4/FP8 conversion).
// Below, the same logic is copied with a different macro name for transparency.

#ifndef CUDA_FP4_FP8_CVT_PTX_SUPPORT
#ifdef __CUDA_ARCH__
#if (CUDART_VERSION >= 12800 && __CUDA_ARCH__ >= 1000 &&
          defined __CUDA_ARCH_FAMILY_SPECIFIC__ &&
          ((__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100) ||
           (__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300)))
#define CUDA_FP4_FP8_CVT_PTX_SUPPORT 1
#else
#define CUDA_FP4_FP8_CVT_PTX_SUPPORT 0
#endif
#else
#define CUDA_FP4_FP8_CVT_PTX_SUPPORT 0
#endif
#endif

template <int bits, int wsize = 8>
inline constexpr __device__ short get_pack_factor() {
            return (bits == 3 || bits == 5) ? 8
                                            : (bits == 6 ? 4 : wsize / bits);
          }

          template <int bits, int wsize = 8>
          inline constexpr __device__ short get_bytes_per_pack() {
            constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
            return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
          }

          template <typename T>
          __device__ __forceinline__ void
          abs_max_x2(T& dst, const T& p1, const T& p2) {
            uint32_t a = *reinterpret_cast<const uint32_t*>(&p1);
            uint32_t b = *reinterpret_cast<const uint32_t*>(&p2);
            uint32_t d;

            if constexpr (std::is_same<T, __nv_bfloat162>::value) {
              asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n"
                           : "=r"(d)
                           : "r"(a), "r"(b));
            } else if constexpr (std::is_same<T, __half2>::value) {
              asm volatile("max.xorsign.abs.f16x2 %0, %1, %2;\n"
                           : "=r"(d)
                           : "r"(a), "r"(b));
            } else if constexpr (std::is_same<T, float2>::value) {
              float2 a = p1;
              float2 b = p2;
              dst.x = fmaxf(fabsf(a.x), fabsf(b.x));
              dst.y = fmaxf(fabsf(a.y), fabsf(b.y));
            }
            *reinterpret_cast<uint32_t*>(&dst) = d;
          }

          } // namespace cu

          template <typename F>
          void dispatch_groups(int group_size, F&& f) {
            switch (group_size) {
              case 32:
                f(std::integral_constant<int, 32>{});
                break;
              case 64:
                f(std::integral_constant<int, 64>{});
                break;
              case 128:
                f(std::integral_constant<int, 128>{});
                break;
            }
          }

          template <typename F>
          void dispatch_bits(int bits, F&& f) {
            switch (bits) {
              case 2:
                f(std::integral_constant<int, 2>{});
                break;
              case 3:
                f(std::integral_constant<int, 3>{});
                break;
              case 4:
                f(std::integral_constant<int, 4>{});
                break;
              case 5:
                f(std::integral_constant<int, 5>{});
                break;
              case 6:
                f(std::integral_constant<int, 6>{});
                break;
              case 8:
                f(std::integral_constant<int, 8>{});
                break;
            }
          }

          } // namespace mlx::core
