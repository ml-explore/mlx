// Copyright © 2025 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

// Batched Cholesky factorization.
//
// The factorization runs in place in `out`. Three paths by matrix size:
//  - N <= 32 (cholesky_simd): one matrix per simdgroup. Lane i keeps row i in
//    registers and lanes communicate with simd_shuffle, so there are no
//    threadgroup barriers at all; several matrices run per threadgroup.
//  - N <= 90 (cholesky_shared): one matrix per threadgroup, staged in
//    threadgroup memory.
//  - larger (cholesky_device): one matrix per threadgroup, factored directly
//    in device memory.

// Right-looking, register-resident factorization for N <= 32. Lane i owns row
// i of the (symmetric) input. After step j every lane i holds the final
// L[i][j] in r[j] (i >= j). The trailing update at step j broadcasts column j
// one element at a time with simd_shuffle; simdgroup lockstep replaces
// barriers.
template <typename T>
[[kernel]] void cholesky_simd(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& upper [[buffer(2)]],
    const constant int& num_matrices [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simds_per_tg [[simdgroups_per_threadgroup]]) {
  int mat = int(tg_id) * int(simds_per_tg) + int(simd_group_id);
  if (mat >= num_matrices) {
    return;
  }
  device T* A = out + size_t(mat) * N * N;
  int lane = int(simd_lane_id);

  // Row `lane` of the matrix (rows equal columns for the symmetric input).
  // For N = 32 each row is exactly one cache line.
  T r[32] = {T(0)};
  if (lane < N) {
    for (int k = 0; k < N; ++k) {
      r[k] = A[lane * N + k];
    }
  }

  for (int j = 0; j < N; ++j) {
    // r[j] on lane j is the fully updated diagonal element.
    T piv = metal::sqrt(simd_shuffle(r[j], ushort(j)));
    if (lane == j) {
      r[j] = piv;
    } else if (lane > j) {
      r[j] /= piv; // L[lane][j]
    }
    // Trailing update: M[i][k] -= L[i][j] * L[k][j] for j < k <= i.
    for (int k = j + 1; k < N; ++k) {
      T lkj = simd_shuffle(r[j], ushort(k));
      if (lane >= k) {
        r[k] -= r[j] * lkj;
      }
    }
  }

  if (lane < N) {
    if (upper == 0) {
      // Row `lane` of L, upper triangle zeroed.
      for (int k = 0; k < N; ++k) {
        A[lane * N + k] = (k <= lane) ? r[k] : T(0);
      }
    } else {
      // R = Lᵀ: lane writes column `lane` (coalesced), lower triangle zeroed.
      for (int k = 0; k < N; ++k) {
        A[k * N + lane] = (k <= lane) ? r[k] : T(0);
      }
    }
  }
}

// In-place factorization bodies. Both produce L (A = L Lᵀ) when !upper or R
// (A = Rᵀ R) when upper, thread `lid` owns the strided rows/cols
// {lid, lid+tgsize, ...} (any threadgroup size <= N is correct), and only the
// relevant triangle + diagonal are written; the caller zeroes the opposite
// strict triangle.
//
// cholesky_core_rl (right-looking) keeps the trailing matrix updated so the
// pivot is a single sqrt; best when M is in threadgroup memory.
// cholesky_core_ll (left-looking) touches only one column/row per step and
// so much less memory; best when M is in device memory.
template <typename T, typename PtrT>
inline void cholesky_core_rl(PtrT M, int N, bool upper, uint lid, uint tgsize) {
  if (!upper) {
    // Column sweep: scale column j, then each thread applies the rank-1
    // update to the columns j+1..i of its rows.
    for (int j = 0; j < N; ++j) {
      if (lid == uint(j % int(tgsize))) {
        M[j * N + j] = metal::sqrt(M[j * N + j]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      T piv = M[j * N + j];
      for (int i = int(lid); i < N; i += int(tgsize)) {
        if (i > j) {
          M[i * N + j] /= piv;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      for (int i = int(lid); i < N; i += int(tgsize)) {
        if (i > j) {
          T lij = M[i * N + j];
          for (int k = j + 1; k <= i; ++k) {
            M[i * N + k] -= lij * M[k * N + j];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
  } else {
    // Row sweep producing the upper factor directly (no transpose): scale row
    // i, then each thread updates rows k > i of the upper triangle it owns.
    for (int i = 0; i < N; ++i) {
      if (lid == uint(i % int(tgsize))) {
        M[i * N + i] = metal::sqrt(M[i * N + i]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      T piv = M[i * N + i];
      for (int j = int(lid); j < N; j += int(tgsize)) {
        if (j > i) {
          M[i * N + j] /= piv;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      for (int k = int(lid); k < N; k += int(tgsize)) {
        if (k > i) {
          T rik = M[i * N + k];
          for (int j = k; j < N; ++j) {
            M[k * N + j] -= rik * M[i * N + j];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
  }
}

template <typename T, typename PtrT>
inline void cholesky_core_ll(PtrT M, int N, bool upper, uint lid, uint tgsize) {
  if (!upper) {
    // Column sweep. Thread owning row j computes the pivot; all threads
    // owning rows i>j fill column j below the diagonal.
    for (int j = 0; j < N; ++j) {
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      if (lid == uint(j % int(tgsize))) {
        T s = M[j * N + j];
        for (int k = 0; k < j; ++k) {
          T v = M[j * N + k];
          s -= v * v;
        }
        M[j * N + j] = metal::sqrt(s);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      T piv = M[j * N + j];
      for (int i = int(lid); i < N; i += int(tgsize)) {
        if (i > j) {
          T s = M[i * N + j];
          for (int k = 0; k < j; ++k) {
            s -= M[i * N + k] * M[j * N + k];
          }
          M[i * N + j] = s / piv;
        }
      }
    }
  } else {
    // Row sweep producing the upper factor directly (no transpose).
    for (int i = 0; i < N; ++i) {
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      if (lid == uint(i % int(tgsize))) {
        T s = M[i * N + i];
        for (int k = 0; k < i; ++k) {
          T v = M[k * N + i];
          s -= v * v;
        }
        M[i * N + i] = metal::sqrt(s);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      T piv = M[i * N + i];
      for (int j = int(lid); j < N; j += int(tgsize)) {
        if (j > i) {
          T s = M[i * N + j];
          for (int k = 0; k < i; ++k) {
            s -= M[k * N + i] * M[k * N + j];
          }
          M[i * N + j] = s / piv;
        }
      }
    }
  }
}

// Mid-size matrices: one per threadgroup, staged in threadgroup memory.
template <typename T>
[[kernel]] void cholesky_shared(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& upper [[buffer(2)]],
    threadgroup T* As [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
  device T* A = out + size_t(gid) * N * N;
  int nn = N * N;
  for (int idx = int(lid); idx < nn; idx += int(tgsize)) {
    As[idx] = A[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  cholesky_core_rl<T>(As, N, upper != 0, lid, tgsize);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int idx = int(lid); idx < nn; idx += int(tgsize)) {
    int r = idx / N;
    int c = idx % N;
    T val;
    if (upper == 0) {
      val = (c <= r) ? As[idx] : T(0);
    } else {
      val = (r <= c) ? As[idx] : T(0);
    }
    A[idx] = val;
  }
}

// Large matrices (N > 128): blocked right-looking factorization following the
// batched-MAGMA design (Dong, Haidar, Tomov & Dongarra, "A Fast Batched
// Cholesky Factorization on a GPU"): the host loops over panels of NB = 32
// columns and encodes one batched *panel* dispatch and one batched *SYRK*
// dispatch per step, with a buffer barrier in between. Separate grids keep
// the trailing update, where nearly all the FLOPs are, at high occupancy
// across the whole batch instead of serializing behind the panel inside a
// single threadgroup. The
// working state is always the lower triangle; an epilogue pass zeroes the
// strict upper triangle, or transposes in place for upper = true.

// Panel step at column p: factor the 32x32 diagonal block (POTF2, simdgroup 0
// with the same register/shuffle method as cholesky_simd), invert it (TRTRI,
// one column per lane), and apply the TRSM to the panel rows below as a GEMM
// against the inverted block on the simdgroup-matrix units (the MAGMA batched
// approach). Each simdgroup owns whole 16-row slices (it reads and writes
// only its own rows, so the in-place update needs no synchronization) and a
// sub-16-row tail falls back to one scalar forward-substitution per
// thread. One threadgroup per matrix.
template <typename T>
[[kernel]] void cholesky_panel(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& p [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slane [[thread_index_in_simdgroup]],
    uint n_sgs [[simdgroups_per_threadgroup]]) {
  constexpr int NB = 32;
  threadgroup T Ldiag[NB * NB];
  threadgroup T Linv[NB * NB];

  device T* A = out + size_t(gid) * N * N;
  int W = metal::min(NB, N - p);

  // Stage the lower part of the diagonal block.
  for (int idx = int(lid); idx < W * W; idx += int(tgsize)) {
    int r = idx / W;
    int c = idx % W;
    Ldiag[r * NB + c] = (c <= r) ? A[size_t(p + r) * N + (p + c)] : T(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // POTF2 on simdgroup 0.
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

  // Write the factored diagonal block back.
  for (int idx = int(lid); idx < W * W; idx += int(tgsize)) {
    int r = idx / W;
    int c = idx % W;
    if (c <= r) {
      A[size_t(p + r) * N + (p + c)] = Ldiag[r * NB + c];
    }
  }

  // Rows below the block exist only when the panel has full width (a narrow
  // last panel reaches N).
  if (W < NB) {
    return;
  }

  // TRTRI: invert the diagonal block, one column per lane of simdgroup 0
  // (forward substitution of L x = e_j; the column is zero above j).
  if (sgid == 0) {
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
    for (int i = 0; i < NB; ++i) {
      Linv[i * NB + j] = x[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // TRSM as a GEMM: X_new = X * Linv^T. Each simdgroup takes 16-row slices
  // (2x4 arrangement of 8x8 fragments), reading X from device and writing the
  // result back in place.
  int first = p + NB;
  int tail = first + ((N - first) / 16) * 16;
  for (int r0 = first + int(sgid) * 16; r0 + 16 <= N; r0 += int(n_sgs) * 16) {
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
  }

  // Scalar forward substitution for the sub-16-row tail.
  for (int r = tail + int(lid); r < N; r += int(tgsize)) {
    T x[NB];
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < NB; ++j) {
      x[j] = A[size_t(r) * N + (p + j)];
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < NB; ++j) {
      T s = x[j];
      MLX_MTL_PRAGMA_UNROLL
      for (int k = 0; k < j; ++k) {
        s -= x[k] * Ldiag[j * NB + k];
      }
      x[j] = s / Ldiag[j * NB + j];
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < NB; ++j) {
      A[size_t(r) * N + (p + j)] = x[j];
    }
  }
}

// Trailing (SYRK) update: C -= L21 * L21^T over the 32x32 lower-triangle
// tiles of the trailing matrix, using the KD panel columns p..p+KD-1 (the
// trailing matrix starts at p+KD). Grid is (lower tiles) x (batch); each
// threadgroup of 4 simdgroups computes one tile as a 2x2 arrangement of 16x16
// simdgroup-matrix products. Boundary tiles (crossing N) take a scalar path.
// With col0 != 0 only the first tile column is updated (tg_pos.x is the tile
// row): that primes the next 32-column panel so that a following KD = 64
// update can process the rest of the trailing with half the C traffic; the
// panels stay 32 wide while the read-modify-write of C halves.
template <typename T, int KD>
[[kernel]] void cholesky_syrk(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& p [[buffer(2)]],
    const constant int& col0 [[buffer(3)]],
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

  // Lower-triangle tile (ti >= tj) from the linear index.
  int t = int(tg_pos.x);
  int ti;
  int tj;
  if (col0 != 0) {
    ti = t;
    tj = 0;
  } else {
    ti = int((metal::sqrt(8.0f * float(t) + 1.0f) - 1.0f) * 0.5f);
    while (ti * (ti + 1) / 2 > t) {
      --ti;
    }
    while ((ti + 1) * (ti + 2) / 2 <= t) {
      ++ti;
    }
    tj = t - ti * (ti + 1) / 2;
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

  if ((ti + 1) * NB <= Ht) {
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
      if (ti * NB + er < Ht && tj * NB + ec < Ht) {
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

// Larger matrices: one per threadgroup, factored directly in device memory.
template <typename T>
[[kernel]] void cholesky_device(
    device T* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& upper [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
  device T* A = out + size_t(gid) * N * N;

  cholesky_core_ll<T>(A, N, upper != 0, lid, tgsize);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  int nn = N * N;
  for (int idx = int(lid); idx < nn; idx += int(tgsize)) {
    int r = idx / N;
    int c = idx % N;
    if (upper == 0) {
      if (c > r) {
        A[idx] = T(0);
      }
    } else {
      if (c < r) {
        A[idx] = T(0);
      }
    }
  }
}
