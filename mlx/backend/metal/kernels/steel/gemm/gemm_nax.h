// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"

using namespace metal;

namespace mlx::steel {

template <
    typename T,
    short SM,
    short SN,
    short SK,
    short BK,
    bool transpose_a,
    bool transpose_b,
    bool kAlignedM,
    bool kAlignedN,
    bool kAlignedK,
    short UM,
    short UN,
    short UK,
    typename AccumType = float>
auto gemm_loop(
    const device T* A,
    const device T* B,
    const constant GEMMParams* params [[buffer(4)]],
    const short sgp_sm,
    const short sgp_sn) {
  constexpr short TM = SM / UM;
  constexpr short TN = SN / UN;
  constexpr short TK = SK / UK;

  constexpr int RA = transpose_a ? TK : TM;
  constexpr int CA = transpose_a ? TM : TK;

  constexpr int RB = transpose_b ? TN : TK;
  constexpr int CB = transpose_b ? TK : TN;

  using DSubTile = NAXSubTile<AccumType, UM, UN>;
  using ASubTile =
      NAXSubTile<T, (transpose_a ? UK : UM), (transpose_a ? UM : UK)>;
  using BSubTile =
      NAXSubTile<T, (transpose_b ? UN : UK), (transpose_b ? UK : UN)>;

  NAXTile<AccumType, TM, TN, DSubTile> Dtile;
  Dtile.clear();

  int gemm_k_iterations_ = params->gemm_k_iterations_aligned;

  STEEL_PRAGMA_NO_UNROLL
  for (int kk0 = 0; kk0 < gemm_k_iterations_; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, RA, CA, ASubTile> Atile;
      NAXTile<T, RB, CB, BSubTile> Btile;
      const int k = kk1;

      volatile int compiler_barrier;

      const int A_offset = transpose_a ? k * params->lda : k;
      const int B_offset = transpose_b ? k : k * params->ldb;

      if constexpr (kAlignedM) {
        Atile.load(A + A_offset, params->lda);
      } else {
        const short rmax = transpose_a ? SK : sgp_sm;
        const short cmax = transpose_a ? sgp_sm : SK;
        Atile.load_safe(A + A_offset, params->lda, short2(cmax, rmax));
      }

      if constexpr (kAlignedN) {
        Btile.load(B + B_offset, params->ldb);
      } else {
        const short rmax = transpose_b ? sgp_sn : SK;
        const short cmax = transpose_b ? SK : sgp_sn;
        Btile.load_safe(B + B_offset, params->ldb, short2(cmax, rmax));
      }

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    A += transpose_a ? (BK * params->lda) : BK;
    B += transpose_b ? BK : (BK * params->ldb);
  }

  if constexpr (!kAlignedK) {
    simdgroup_barrier(mem_flags::mem_none);

    const short rem_bk = params->K - gemm_k_iterations_ * BK;

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
      NAXTile<T, 1, 1, ASubTile> Atile;
      NAXTile<T, 1, 1, BSubTile> Btile;

      STEEL_PRAGMA_UNROLL
      for (int mm = 0; mm < TM; mm++) {
        STEEL_PRAGMA_UNROLL
        for (int nn = 0; nn < TN; nn++) {
          STEEL_PRAGMA_UNROLL
          for (int kk = 0; kk < TK; kk++) {
            const int m = mm * UM;
            const int n = nn * UN;
            const int k = kk1 + kk * UK;
            const short psk = max(0, rem_bk - k);

            const int A_offset =
                transpose_a ? (m + k * params->lda) : (m * params->lda + k);
            const int B_offset =
                transpose_b ? (k + n * params->ldb) : (k * params->ldb + n);

            {
              const short psm = kAlignedM ? SM : max(0, sgp_sm - m);
              const short rmax = transpose_a ? psk : psm;
              const short cmax = transpose_a ? psm : psk;
              Atile.load_safe(A + A_offset, params->lda, short2(cmax, rmax));
            }

            {
              const short psn = kAlignedN ? SN : max(0, sgp_sn - n);
              const short rmax = transpose_b ? psn : psk;
              const short cmax = transpose_b ? psk : psn;
              Btile.load_safe(B + B_offset, params->ldb, short2(cmax, rmax));
            }

            subtile_matmad_nax(
                Dtile.subtile_at(mm, nn),
                Atile.subtile_at(0, 0),
                metal::bool_constant<transpose_a>{},
                Btile.subtile_at(0, 0),
                metal::bool_constant<transpose_b>{});
          }
        }
      }
    }
  }

  return Dtile;
}

} // namespace mlx::steel
