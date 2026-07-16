// Copyright © 2025 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

// Cholesky factorization kernels.
//
// The host (backend/metal/cholesky.cpp) factors the matrix in place in `out`
// with a blocked right-looking algorithm: strips of 512 columns are factored
// 32 columns at a time (cholesky_potf2 + cholesky_trsm + a strip-bounded
// cholesky_syrk), the large trailing update runs on the steel GEMM, and a
// final cholesky_fixup pass zeroes the strict upper triangle (or transposes
// in place for the upper factor). Only the lower triangle is read or written
// until the fixup pass, so the same kernels serve both triangles.
//
// The three phases are separate dispatches ordered by buffer barriers: every
// location is written by at most one dispatch and read by later ones, so
// there are no cross-threadgroup hazards inside any dispatch.

// Factor the (up to) 32x32 diagonal block at (p, p). One threadgroup per
// matrix: simdgroup 0 keeps one row per lane in registers and broadcasts the
// pivot column with simd_shuffle, writes the factored block back, and stores
// the inverse of the block (TRTRI, one column per lane) to `linv` for the
// TRSM dispatch that follows.
template <typename T>
[[kernel]] void cholesky_potf2(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& p [[buffer(2)]],
    device T* linv [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slane [[thread_index_in_simdgroup]]) {
  constexpr int NB = 32;
  threadgroup T Ldiag[NB * NB];

  device T* A = out + size_t(gid) * N * N;
  int W = metal::min(NB, N - p);

  for (int idx = int(lid); idx < W * W; idx += int(tgsize)) {
    int r = idx / W;
    int c = idx % W;
    Ldiag[r * NB + c] = (c <= r) ? A[size_t(p + r) * N + (p + c)] : T(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Right-looking factorization in registers: after step j every lane i
  // holds the final L[i][j] in r[j] (i >= j). Lanes run in lockstep within
  // the simdgroup, so no barriers are needed.
  if (sgid == 0) {
    int lane = int(slane);
    T r[NB] = {T(0)};
    if (lane < W) {
      for (int k = 0; k <= lane; ++k) {
        r[k] = Ldiag[lane * NB + k];
      }
    }
    for (int j = 0; j < W; ++j) {
      T piv = metal::sqrt(simd_shuffle(r[j], ushort(j)));
      if (lane == j) {
        r[j] = piv;
      } else if (lane > j) {
        r[j] /= piv;
      }
      for (int k = j + 1; k < W; ++k) {
        T lkj = simd_shuffle(r[j], ushort(k));
        if (lane >= k) {
          r[k] -= r[j] * lkj;
        }
      }
    }
    if (lane < W) {
      for (int k = 0; k <= lane; ++k) {
        Ldiag[lane * NB + k] = r[k];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int idx = int(lid); idx < W * W; idx += int(tgsize)) {
    int r = idx / W;
    int c = idx % W;
    if (c <= r) {
      A[size_t(p + r) * N + (p + c)] = Ldiag[r * NB + c];
    }
  }

  // TRTRI for the TRSM below the block (only needed when such rows exist).
  if (N - p > NB && sgid == 0) {
    int j = int(slane);
    T x[NB] = {T(0)};
    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < NB; ++i) {
      T s = (i == j) ? T(1) : T(0);
      MLX_MTL_PRAGMA_UNROLL
      for (int k = 0; k < i; ++k) {
        s -= Ldiag[i * NB + k] * x[k];
      }
      x[i] = (i >= j) ? s / Ldiag[i * NB + i] : T(0);
    }
    device T* lv = linv + size_t(gid) * NB * NB;
    for (int i = 0; i < NB; ++i) {
      lv[i * NB + j] = x[i];
    }
  }
}

// TRSM: X_new = X * Linv^T for the panel rows below the diagonal block,
// 128 rows per threadgroup (grid is row blocks x batch). Each simdgroup owns
// one 16-row slice as a 2x4 arrangement of 8x8 simdgroup-matrix fragments;
// rows past the last full slice fall back to one scalar product per lane.
template <typename T>
[[kernel]] void cholesky_trsm(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& p [[buffer(2)]],
    const device T* linv [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 tgsize2 [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slane [[thread_index_in_simdgroup]]) {
  constexpr int NB = 32;
  threadgroup T Linv[NB * NB];

  uint lid = lid2.x;
  uint tgsize = tgsize2.x;
  device T* A = out + size_t(tg_pos.y) * N * N;
  const device T* lv = linv + size_t(tg_pos.y) * NB * NB;

  for (int idx = int(lid); idx < NB * NB; idx += int(tgsize)) {
    Linv[idx] = lv[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int r0 = p + NB + int(tg_pos.x) * 128 + int(sgid) * 16;
  if (r0 + 16 <= N) {
    metal::simdgroup_matrix<T, 8, 8> acc[2][4];
    MLX_MTL_PRAGMA_UNROLL
    for (int ra = 0; ra < 2; ++ra) {
      MLX_MTL_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c) {
        acc[ra][c] = metal::simdgroup_matrix<T, 8, 8>(T(0));
      }
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int kb = 0; kb < NB; kb += 8) {
      metal::simdgroup_matrix<T, 8, 8> a[2];
      MLX_MTL_PRAGMA_UNROLL
      for (int ra = 0; ra < 2; ++ra) {
        simdgroup_load(a[ra], A + size_t(r0 + ra * 8) * N + (p + kb), ulong(N));
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c) {
        metal::simdgroup_matrix<T, 8, 8> b;
        simdgroup_load(
            b, Linv + c * 8 * NB + kb, ulong(NB), ulong2(0, 0), true);
        MLX_MTL_PRAGMA_UNROLL
        for (int ra = 0; ra < 2; ++ra) {
          simdgroup_multiply_accumulate(acc[ra][c], a[ra], b, acc[ra][c]);
        }
      }
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int ra = 0; ra < 2; ++ra) {
      MLX_MTL_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c) {
        simdgroup_store(
            acc[ra][c], A + size_t(r0 + ra * 8) * N + (p + c * 8), ulong(N));
      }
    }
  } else if (r0 < N) {
    // Scalar tail: each lane handles one row.
    int r = r0 + int(slane);
    if (r < N) {
      T x[NB];
      MLX_MTL_PRAGMA_UNROLL
      for (int k = 0; k < NB; ++k) {
        x[k] = A[size_t(r) * N + (p + k)];
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < NB; ++j) {
        T s = T(0);
        MLX_MTL_PRAGMA_UNROLL
        for (int k = 0; k < NB; ++k) {
          s += x[k] * Linv[j * NB + k];
        }
        A[size_t(r) * N + (p + j)] = s;
      }
    }
  }
}

// Trailing (SYRK) update: C -= L21 * L21^T over 32x32 tiles, using the KD
// panel columns p..p+KD-1 (the trailing matrix starts at q = p + KD). Only
// the first ncols trailing columns are updated: the host bounds the update
// to the current strip and leaves the rest of the trailing matrix to the
// steel GEMM. The grid is (row tiles x column tiles) x batch; each
// threadgroup of 4 simdgroups computes one tile as a 2x2 arrangement of
// 16x16 simdgroup-matrix products, and tiles crossing the row or column
// bound take a scalar path.
template <typename T, int KD>
[[kernel]] void cholesky_syrk(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& p [[buffer(2)]],
    const constant int& ncols [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 tgsize2 [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
  constexpr int NB = 32;
  threadgroup T As[NB * KD];
  threadgroup T Bs[NB * KD];

  uint lid = lid2.x;
  uint tgsize = tgsize2.x;
  device T* A = out + size_t(tg_pos.y) * N * N;
  int q = p + KD;
  int Ht = N - q;

  // Tiles above the diagonal have nothing to update; the whole threadgroup
  // returns before the barrier below.
  int ntj = (ncols + NB - 1) / NB;
  int t = int(tg_pos.x);
  int ti = t / ntj;
  int tj = t % ntj;
  if (ti < tj) {
    return;
  }

  // Stage the two panel blocks this tile needs (rows q + ti*32 / q + tj*32 of
  // panel columns p..p+KD-1), zero-padded past N.
  for (int idx = int(lid); idx < NB * KD; idx += int(tgsize)) {
    int r = idx / KD;
    int c = idx % KD;
    int ra = q + ti * NB + r;
    int rb = q + tj * NB + r;
    As[idx] = (ra < N) ? A[size_t(ra) * N + (p + c)] : T(0);
    Bs[idx] = (rb < N) ? A[size_t(rb) * N + (p + c)] : T(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if ((ti + 1) * NB <= Ht && (tj + 1) * NB <= ncols) {
    // Full tile: 4 simdgroups, each a 16x16 block of the 32x32 tile.
    int sr = int(sgid) / 2;
    int sc = int(sgid) % 2;
    metal::simdgroup_matrix<T, 8, 8> acc[2][2];
    MLX_MTL_PRAGMA_UNROLL
    for (int fr = 0; fr < 2; ++fr) {
      MLX_MTL_PRAGMA_UNROLL
      for (int fc = 0; fc < 2; ++fc) {
        acc[fr][fc] = metal::simdgroup_matrix<T, 8, 8>(T(0));
      }
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int kb = 0; kb < KD; kb += 8) {
      metal::simdgroup_matrix<T, 8, 8> a[2];
      metal::simdgroup_matrix<T, 8, 8> b[2];
      MLX_MTL_PRAGMA_UNROLL
      for (int f = 0; f < 2; ++f) {
        simdgroup_load(a[f], As + (sr * 16 + f * 8) * KD + kb, ulong(KD));
        simdgroup_load(
            b[f],
            Bs + (sc * 16 + f * 8) * KD + kb,
            ulong(KD),
            ulong2(0, 0),
            true);
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int fr = 0; fr < 2; ++fr) {
        MLX_MTL_PRAGMA_UNROLL
        for (int fc = 0; fc < 2; ++fc) {
          simdgroup_multiply_accumulate(acc[fr][fc], a[fr], b[fc], acc[fr][fc]);
        }
      }
    }
    device T* C =
        A + size_t(q + ti * NB + sr * 16) * N + (q + tj * NB + sc * 16);
    MLX_MTL_PRAGMA_UNROLL
    for (int fr = 0; fr < 2; ++fr) {
      MLX_MTL_PRAGMA_UNROLL
      for (int fc = 0; fc < 2; ++fc) {
        device T* Cf = C + size_t(fr) * 8 * N + fc * 8;
        metal::simdgroup_matrix<T, 8, 8> c;
        simdgroup_load(c, Cf, ulong(N));
        c.thread_elements()[0] -= acc[fr][fc].thread_elements()[0];
        c.thread_elements()[1] -= acc[fr][fc].thread_elements()[1];
        simdgroup_store(c, Cf, ulong(N));
      }
    }
  } else {
    // Boundary tile: scalar with bounds checks.
    for (int e = int(lid); e < NB * NB; e += int(tgsize)) {
      int er = e / NB;
      int ec = e % NB;
      if (ti * NB + er < Ht && tj * NB + ec < ncols) {
        T s = T(0);
        for (int k = 0; k < KD; ++k) {
          s += As[er * KD + k] * Bs[ec * KD + k];
        }
        A[size_t(q + ti * NB + er) * N + (q + tj * NB + ec)] -= s;
      }
    }
  }
}

// Epilogue: the factor lives in the lower triangle. Zero the strict upper
// triangle, or for upper = true move it in place (R = L^T) and zero the
// strict lower. One thread per (matrix, row): each thread streams contiguous
// elements of its own row, and for upper only ever writes the strict upper
// triangle while reading its own lower row, so threads never collide.
template <typename T>
[[kernel]] void cholesky_fixup(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& upper [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
  int r = int(pos.x);
  device T* A = out + size_t(pos.y) * N * N;
  if (upper == 0) {
    for (int c = r + 1; c < N; ++c) {
      A[size_t(r) * N + c] = T(0);
    }
  } else {
    for (int c = 0; c < r; ++c) {
      A[size_t(c) * N + r] = A[size_t(r) * N + c];
      A[size_t(r) * N + c] = T(0);
    }
  }
}
