// Copyright © 2025 Apple Inc.
#pragma once

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "mlx/backend/cpu/gemms/aligned_buffer.h"
#include "mlx/backend/cpu/gemms/avx_gemm_simd.h"
#include "mlx/backend/cpu/gemms/avx_simd_gemv.h"

namespace mlx::core {

template <typename T>
inline void
pack_transpose_8x8(const T* src, float* dst, int src_stride, int dst_stride) {
  detail::transpose_8x8_block<T>(src, dst, src_stride, dst_stride);
}

// Pack A block (m_block x k_block) into A_packed (MC x KC float, column-major).
template <typename T, int MC, int KC>
static void pack_A_block(
    const T* A,
    float* A_packed,
    int M,
    int K,
    int ldA,
    int M_offset,
    int K_offset,
    int m_block,
    int k_block,
    bool a_trans) {
  static_assert(
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      "T must be float16 or bfloat16");
  constexpr int simd_width = 8;

  // Zero-fill only the portions we access (edge tiles)
  if (m_block < MC || k_block < KC) {
    for (int k = 0; k < k_block; ++k) {
      std::fill(A_packed + k * MC, A_packed + k * MC + m_block, 0.0f);
    }
  }

  if (!a_trans) {
    // A is row-major (M x K). Pack with 8x8 transpose blocks.
    for (int k = 0; k < k_block; k += 8) {
      int k_chunk = std::min(8, k_block - k);

      if (k_chunk == 8) {
        for (int i = 0; i < m_block; i += 8) {
          int m_chunk = std::min(8, m_block - i);

          if (m_chunk == 8) {
            const T* a_block_start = A + (M_offset + i) * ldA + K_offset + k;
            pack_transpose_8x8<T>(
                a_block_start, A_packed + k * MC + i, ldA, MC);
          } else {
            for (int ii = 0; ii < m_chunk; ++ii) {
              const T* a_src_row_ptr =
                  A + (M_offset + i + ii) * ldA + K_offset + k;
              for (int kk = 0; kk < k_chunk; ++kk) {
                A_packed[(k + kk) * MC + (i + ii)] =
                    static_cast<float>(a_src_row_ptr[kk]);
              }
            }
          }
        }
      } else {
        for (int i = 0; i < m_block; ++i) {
          const T* a_src_row_ptr = A + (M_offset + i) * ldA + K_offset + k;
          for (int kk = 0; kk < k_chunk; ++kk) {
            A_packed[(k + kk) * MC + i] = static_cast<float>(a_src_row_ptr[kk]);
          }
        }
      }
    }
  } else {
    // A is transposed (K x M row-major). Contiguous copy with SIMD convert.
    for (int k = 0; k < k_block; ++k) {
      const T* a_src_row_ptr = A + (K_offset + k) * ldA + M_offset;
      float* a_dst_col_ptr = A_packed + k * MC;
      int i = 0;
      for (; i + simd_width <= m_block; i += simd_width) {
        detail::Simd<float, 8> a_vec =
            detail::load_convert_to_float<T>(a_src_row_ptr + i);
        detail::store<float, simd_width>(a_dst_col_ptr + i, a_vec);
      }
      for (; i < m_block; ++i) {
        a_dst_col_ptr[i] = static_cast<float>(a_src_row_ptr[i]);
      }
    }
  }
}

// Pack B block (k_block x n_block) into B_packed (KC x NC float, row-major).
template <typename T, int KC, int NC>
static void pack_B_block(
    const T* B,
    float* B_packed,
    int K,
    int N,
    int ldB,
    int K_offset,
    int N_offset,
    int k_block,
    int n_block,
    bool b_trans) {
  static_assert(
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      "T must be float16 or bfloat16");
  constexpr int simd_width = 8;

  if (k_block < KC || n_block < NC) {
    for (int k = 0; k < k_block; ++k) {
      std::fill(B_packed + k * NC, B_packed + k * NC + n_block, 0.0f);
    }
  }

  if (!b_trans) {
    // B is row-major (K x N). Contiguous copy with SIMD convert.
    for (int k = 0; k < k_block; ++k) {
      const T* b_src_row_ptr = B + (K_offset + k) * ldB + N_offset;
      float* b_dst_row_ptr = B_packed + k * NC;
      int j = 0;
      for (; j + simd_width <= n_block; j += simd_width) {
        detail::Simd<float, 8> b_vec =
            detail::load_convert_to_float<T>(b_src_row_ptr + j);
        detail::store<float, simd_width>(b_dst_row_ptr + j, b_vec);
      }
      for (; j < n_block; ++j) {
        b_dst_row_ptr[j] = static_cast<float>(b_src_row_ptr[j]);
      }
    }
  } else {
    // B is transposed (N x K row-major). Pack with 8x8 transpose blocks.
    for (int k = 0; k < k_block; k += 8) {
      int k_chunk = std::min(8, k_block - k);

      if (k_chunk == 8) {
        for (int j = 0; j < n_block; j += 8) {
          int n_chunk = std::min(8, n_block - j);

          if (n_chunk == 8) {
            const T* b_block_start = B + (N_offset + j) * ldB + K_offset + k;
            float tmp_transpose[64];
            pack_transpose_8x8<T>(b_block_start, tmp_transpose, ldB, 8);
            for (int kk = 0; kk < 8; ++kk) {
              for (int jj = 0; jj < 8; ++jj) {
                B_packed[(k + kk) * NC + (j + jj)] = tmp_transpose[kk * 8 + jj];
              }
            }
          } else {
            for (int kk = 0; kk < k_chunk; ++kk) {
              float* b_dst_row_ptr = B_packed + (k + kk) * NC + j;
              for (int jj = 0; jj < n_chunk; ++jj) {
                b_dst_row_ptr[jj] = static_cast<float>(
                    B[(N_offset + j + jj) * ldB + (K_offset + k + kk)]);
              }
            }
          }
        }
      } else {
        for (int kk = 0; kk < k_chunk; ++kk) {
          float* b_dst_row_ptr = B_packed + (k + kk) * NC;
          for (int j = 0; j < n_block; ++j) {
            b_dst_row_ptr[j] = static_cast<float>(
                B[(N_offset + j) * ldB + (K_offset + k + kk)]);
          }
        }
      }
    }
  }
}

// Single-threaded fp16/bf16 GEMM with fp32 accumulation. Goto-style
// jc→ic→pc blocking; A and B are packed to fp32 once per panel.
template <typename T>
void simd_gemm_optimized_higher_precision(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    int ldA,
    int ldB,
    int ldC,
    float alpha,
    float beta) {
  static_assert(
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      "GEMM kernel requires float16_t or bfloat16_t.");

  // Blocking parameters.
  constexpr int MR = 6;
  constexpr int NR = 16;
  static_assert(NR % 8 == 0, "NR must be multiple of float SIMD width (8)");

  constexpr int KC_BLOCK = 256;
  constexpr int MC_BLOCK = 96;
  constexpr int NC_BLOCK = 256;

  static_assert(MC_BLOCK % MR == 0, "MC_BLOCK must be a multiple of MR");
  static_assert(NC_BLOCK % NR == 0, "NC_BLOCK must be a multiple of NR");

  // Fixed-size scratch routed through MLX's allocator. The MC x NC
  // accumulator stays bounded because pc is nested inside ic below.
  aligned_scratch A_scratch(MC_BLOCK * KC_BLOCK);
  aligned_scratch B_scratch(KC_BLOCK * NC_BLOCK);
  aligned_scratch C_scratch(MC_BLOCK * NC_BLOCK);

  float* A_packed = A_scratch.get();
  float* B_packed = B_scratch.get();
  float* C_acc = C_scratch.get();

  // Scalar fallback for edge tiles (m_micro < MR or n_micro < NR).
  auto compute_block_scalar_partial = [](

                                          const float* A_panel,
                                          const float* B_panel,
                                          float* C_sub,
                                          int ldc_acc,
                                          int m_micro,
                                          int n_micro,
                                          int k_block,
                                          int a_stride,
                                          int b_stride) {
    for (int i = 0; i < m_micro; ++i) {
      for (int j = 0; j < n_micro; ++j) {
        float acc = C_sub[i * ldc_acc + j];
        for (int k = 0; k < k_block; ++k) {
          acc += A_panel[i + k * a_stride] * B_panel[k * b_stride + j];
        }
        C_sub[i * ldc_acc + j] = acc;
      }
    }
  };

  constexpr int sw = 8;

  for (int jc = 0; jc < N; jc += NC_BLOCK) {
    int nc = std::min(NC_BLOCK, N - jc);

    for (int ic = 0; ic < M; ic += MC_BLOCK) {
      int mc = std::min(MC_BLOCK, M - ic);

      // Zero the accumulator; it persists across pc so accumulation stays
      // in fp32. alpha and beta*C are applied at writeback.
      for (int i = 0; i < mc; ++i) {
        std::memset(C_acc + i * NC_BLOCK, 0, nc * sizeof(float));
      }

      for (int pc = 0; pc < K; pc += KC_BLOCK) {
        int kc = std::min(KC_BLOCK, K - pc);

        // B is re-packed per ic block to keep the accumulator bounded.
        pack_B_block<T, KC_BLOCK, NC_BLOCK>(
            b, B_packed, K, N, ldB, pc, jc, kc, nc, b_trans);
        pack_A_block<T, MC_BLOCK, KC_BLOCK>(
            a, A_packed, M, K, ldA, ic, pc, mc, kc, a_trans);

        // Microkernel loop
        for (int ir = 0; ir < mc; ir += MR) {
          int m_micro = std::min(MR, mc - ir);

          for (int jr = 0; jr < nc; jr += NR) {
            int n_micro = std::min(NR, nc - jr);

            const float* a_ptr = A_packed + ir;
            const float* b_ptr = B_packed + jr;
            float* c_ptr = C_acc + ir * NC_BLOCK + jr;

            // Prefetch next C_acc tile into L2
            if (jr + NR < nc) {
              for (int pi = 0; pi < MR && ir + pi < mc; ++pi)
                _mm_prefetch(
                    reinterpret_cast<const char*>(
                        C_acc + (ir + pi) * NC_BLOCK + jr + NR),
                    _MM_HINT_T1);
            } else if (ir + MR < mc) {
              for (int pi = 0; pi < MR && ir + MR + pi < mc; ++pi)
                _mm_prefetch(
                    reinterpret_cast<const char*>(
                        C_acc + (ir + MR + pi) * NC_BLOCK),
                    _MM_HINT_T1);
            }

            if (m_micro == MR && n_micro == NR) {
              detail::micro_kernel_6x16(
                  a_ptr, b_ptr, c_ptr, NC_BLOCK, kc, MC_BLOCK, NC_BLOCK);
            } else {
              compute_block_scalar_partial(
                  a_ptr,
                  b_ptr,
                  c_ptr,
                  NC_BLOCK,
                  m_micro,
                  n_micro,
                  kc,
                  MC_BLOCK,
                  NC_BLOCK);
            }
          }
        }
      } // pc

      // Writeback: C = alpha * acc + beta * C
      {
        bool apply_alpha = (alpha != 1.0f);
        bool apply_beta = (beta != 0.0f);
        detail::Simd<float, 8> alpha_vec(alpha);
        detail::Simd<float, 8> beta_vec(beta);

        for (int i = 0; i < mc; ++i) {
          T* c_row = c + (ic + i) * ldC + jc;
          float* acc_row = C_acc + i * NC_BLOCK;
          int j = 0;
          for (; j + sw <= nc; j += sw) {
            detail::Simd<float, 8> acc = detail::load<float, 8>(acc_row + j);
            if (apply_alpha)
              acc = alpha_vec * acc;
            if (apply_beta) {
              detail::Simd<float, 8> cv =
                  detail::load_convert_to_float<T>(c_row + j);
              acc = acc + beta_vec * cv;
            }
            detail::store_convert_from_float<T>(c_row + j, acc);
          }
          for (; j < nc; ++j) {
            float val = acc_row[j];
            if (apply_alpha)
              val *= alpha;
            if (apply_beta)
              val += beta * static_cast<float>(c_row[j]);
            c_row[j] = static_cast<T>(val);
          }
        }
      }
    } // ic
  } // jc
}

// Public interface: validates dimensions and dispatches to the blocked kernel.
template <typename T, typename AccT>
void simd_gemm(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    size_t M_s,
    size_t N_s,
    size_t K_s,
    float alpha = 1.0f,
    float beta = 0.0f) {
  static_assert(
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      "simd_gemm requires T = float16_t or bfloat16_t.");
  static_assert(
      std::is_same_v<AccT, float>, "simd_gemm requires AccT = float.");

  if (M_s > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      N_s > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      K_s > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Matrix dimensions exceed int limits.");
  }
  int M = static_cast<int>(M_s);
  int N = static_cast<int>(N_s);
  int K = static_cast<int>(K_s);

  if (M <= 0 || N <= 0)
    return;

  int ldA = (!a_trans) ? K : M;
  int ldB = (!b_trans) ? N : K;
  int ldC = N;

  // K=0: C = beta * C
  if (K <= 0) {
    if (beta == 0.0f) {
      for (int i = 0; i < M; ++i) {
        T zero_val = static_cast<T>(0.0f);
        std::fill(c + i * ldC, c + i * ldC + N, zero_val);
      }
    } else if (beta != 1.0f) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          float c_old_f = static_cast<float>(c[i * ldC + j]);
          c[i * ldC + j] = static_cast<T>(beta * c_old_f);
        }
      }
    }
    return;
  }

  // Dispatch to GEMV for M=1 or N=1 (avoids blocked GEMM overhead)
  if (M == 1 || N == 1) {
    simd_gemv<T>(
        a, b, c, a_trans, b_trans, M, N, K, ldA, ldB, ldC, alpha, beta);
    return;
  }

  simd_gemm_optimized_higher_precision<T>(
      a, b, c, a_trans, b_trans, M, N, K, ldA, ldB, ldC, alpha, beta);
}

} // namespace mlx::core