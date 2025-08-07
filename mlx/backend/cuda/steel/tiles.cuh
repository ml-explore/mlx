// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/steel/utils.cuh"

namespace mlx::core::cu {

// Map types to their vector of 2 type float -> float2, double -> double2 etc
template <typename T>
struct Vector2;
template <>
struct Vector2<double> {
  using type = double2;
};
template <>
struct Vector2<float> {
  using type = float2;
};
template <>
struct Vector2<__half> {
  using type = __half2;
};
template <>
struct Vector2<__nv_bfloat16> {
  using type = __nv_bfloat162;
};
template <typename T>
using Vector2_t = typename Vector2<T>::type;

/**
 * The basic building block for Ampere mmas. A 16x16 tile distributed across
 * the warp.
 *
 * Each thread holds 8 values. They are distributed according to
 * https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
 *
 * For use instructions see the individual methods eg load().
 */
template <typename T>
struct Tile16x16 {
  using T2 = Vector2_t<T>;

  T2 values[4];

  __device__ inline void fill(T v) {
    T2 v2 = {v, v};
    for (int i = 0; i < 4; i++) {
      values[i] = v2;
    }
  }

  /**
   * Load a 16x16 tile from shared memory.
   *
   * The instruction is a bit weird in the sense that the address provided by
   * each thread and the elements loaded are not the same.
   *
   * We load 4 8x8 tiles. The tile rows are stored contiguously in memory. As a
   * result the warp provides 4*8 = 32 addresses one per row.
   *
   * Threads 0-7 provide the addresses for the first tile, 8-15 for the second
   * and so on. For instance to load a non swizzled tile we would do
   *
   *    base_addr + (laneid % 16) * BK + (laneid / 2) * 8
   *
   * See
   * https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix
   */
  __device__ __forceinline__ void load(uint32_t row_address) {
    if constexpr (
        std::is_same_v<T2, __nv_bfloat162> || std::is_same_v<T2, __half2>) {
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(*(uint32_t*)&(values[0])),
            "=r"(*(uint32_t*)&(values[1])),
            "=r"(*(uint32_t*)&(values[2])),
            "=r"(*(uint32_t*)&(values[3]))
          : "r"(row_address));
    }
  }

  /**
   * Store the tile to the address pointed to by `x`.
   *
   * The provided pointer is a generic pointer but this is meant to be used to
   * store to global memory. For storing to shared memory we should use
   * `stmatrix`.
   *
   * This also showcases the format of the tile quite nicely. Each register is
   * holding to adjacent values. The indices are
   *
   *    row + 0, col + 0
   *    row + 8, col + 0
   *    row + 0, col + 8
   *    row + 8, col + 8
   *
   * Given that we are dealing with Vector2_t<U> the column offsets are 4
   * instead of 8.
   */
  template <typename U>
  __device__ inline void store_global(U* x, int N) {
    using U2 = Vector2_t<U>;
    U2* x2 = reinterpret_cast<U2*>(x);
    const int laneid = threadIdx.x % 32;
    const int row = laneid / 4;
    const int col = laneid % 4;
    if constexpr (std::is_same_v<U2, T2>) {
      x2[(row + 0) * (N / 2) + col + 0] = values[0];
      x2[(row + 0) * (N / 2) + col + 4] = values[2];
      x2[(row + 8) * (N / 2) + col + 0] = values[1];
      x2[(row + 8) * (N / 2) + col + 4] = values[3];
    } else if constexpr (
        std::is_same_v<T2, float2> && std::is_same_v<U, __nv_bfloat16>) {
      x2[(row + 0) * (N / 2) + col + 0] =
          __floats2bfloat162_rn(values[0].x, values[0].y);
      x2[(row + 0) * (N / 2) + col + 4] =
          __floats2bfloat162_rn(values[2].x, values[2].y);
      x2[(row + 8) * (N / 2) + col + 0] =
          __floats2bfloat162_rn(values[1].x, values[1].y);
      x2[(row + 8) * (N / 2) + col + 4] =
          __floats2bfloat162_rn(values[3].x, values[3].y);
    }
  }

  template <typename U>
  __device__ inline void store_global_safe(U* x, int N, int max_rows) {
    const int laneid = threadIdx.x % 32;
    const int row = laneid / 4;
    const int col = laneid % 4;
    if (row < max_rows) {
      x[(row + 0) * N + 2 * col + 0] = static_cast<U>(values[0].x);
      x[(row + 0) * N + 2 * col + 1] = static_cast<U>(values[0].y);
      x[(row + 0) * N + 2 * col + 8] = static_cast<U>(values[2].x);
      x[(row + 0) * N + 2 * col + 9] = static_cast<U>(values[2].y);
    }
    if (row + 8 < max_rows) {
      x[(row + 8) * N + 2 * col + 0] = static_cast<U>(values[1].x);
      x[(row + 8) * N + 2 * col + 1] = static_cast<U>(values[1].y);
      x[(row + 8) * N + 2 * col + 8] = static_cast<U>(values[3].x);
      x[(row + 8) * N + 2 * col + 9] = static_cast<U>(values[3].y);
    }
  }
};

/**
 * A simple container of multiple Tile16x16.
 *
 * Provides utility functions for loading and manipulating collections of basic
 * tiles.
 */
template <typename T, int ROWS_, int COLS_>
struct RegisterTile {
  static constexpr int ROWS = ROWS_;
  static constexpr int COLS = COLS_;
  static constexpr int TILES_X = COLS / 16;
  static constexpr int TILES_Y = ROWS / 16;

  Tile16x16<T> data[TILES_X * TILES_Y];

  __device__ inline void fill(T v) {
    MLX_UNROLL
    for (int i = 0; i < TILES_Y; i++) {
      MLX_UNROLL
      for (int j = 0; j < TILES_X; j++) {
        data[i * TILES_X + j].fill(v);
      }
    }
  }

  template <typename Tile>
  __device__ __forceinline__ void
  load(Tile& tile, uint32_t base_address, int row, int col) {
    MLX_UNROLL
    for (int i = 0; i < TILES_Y; i++) {
      MLX_UNROLL
      for (int j = 0; j < TILES_X; j++) {
        data[i * TILES_X + j].load(
            tile.loc(base_address, row + i * 16, col + j * 16));
      }
    }
  }

  template <typename Tile, typename F>
  __device__ __forceinline__ void
  load(Tile& tile, F f, uint32_t base_address, int row, int col) {
    MLX_UNROLL
    for (int i = 0; i < TILES_Y; i++) {
      MLX_UNROLL
      for (int j = 0; j < TILES_X; j++) {
        f(data[i * TILES_X + j],
          tile,
          base_address,
          row + i * 16,
          col + j * 16);
      }
    }
  }

  template <typename U>
  __device__ inline void store_global(U* x, int N, int row, int col) {
    MLX_UNROLL
    for (int i = 0; i < TILES_Y; i++) {
      MLX_UNROLL
      for (int j = 0; j < TILES_X; j++) {
        data[i * TILES_X + j].store_global(
            x + (row + i * 16) * N + col + j * 16, N);
      }
    }
  }

  template <typename U>
  __device__ inline void
  store_global_safe(U* x, int N, int row, int col, int max_rows) {
    MLX_UNROLL
    for (int i = 0; i < TILES_Y; i++) {
      MLX_UNROLL
      for (int j = 0; j < TILES_X; j++) {
        data[i * TILES_X + j].store_global_safe(
            x + (row + i * 16) * N + col + j * 16, N, max_rows - row - i * 16);
      }
    }
  }
};

template <typename T, int ROWS_, int COLS_>
struct SharedTile {
  static constexpr int ROWS = ROWS_;
  static constexpr int COLS = COLS_;
  static constexpr int TILES_X = COLS / 16;
  static constexpr int TILES_Y = ROWS / 16;
  static constexpr int NUMEL = ROWS * COLS;

  // Swizzle taken from ThunderKittens. Should be changed when we switch to
  // cute Layouts.
  //
  // See inludes/types/shared/st.cuh
  //
  // I do feel that it is too math heavy and can be improved. Also the math is
  // done every time although the addresses don't change from load to load. I
  // guess we are expecting the compiler to figure that out.
  static constexpr int swizzle_bytes =
      (sizeof(T) == 2 ? (TILES_X % 4 == 0 ? 128 : (TILES_X % 2 == 0 ? 64 : 32))
                      : (sizeof(T) == 4 ? (TILES_X % 2 == 0 ? 128 : 64) : 0));

  T data[ROWS * COLS];

  __device__ inline uint32_t base_addr() const {
    return __cvta_generic_to_shared(&data[0]);
  }

  // Return a pointer to the element at (row, col) using the swizzle.
  __device__ static inline T* ptr(T* ptr, int row, int col) {
    if constexpr (swizzle_bytes > 0) {
      static constexpr int swizzle_repeat = swizzle_bytes * 8;
      static constexpr int subtile_cols = swizzle_bytes / sizeof(T);
      const int outer_idx = col / subtile_cols;
      const uint64_t addr =
          (uint64_t)(&ptr
                         [outer_idx * ROWS * subtile_cols + row * subtile_cols +
                          col % subtile_cols]);
      const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
      return (T*)(addr ^ swizzle);
    } else {
      return ptr + row * COLS + col;
    }
  }

  // Return the location of the element at (row, col) using the swizzle.
  __device__ static inline uint32_t loc(uint32_t ptr, int row, int col) {
    if constexpr (swizzle_bytes > 0) {
      static constexpr int swizzle_repeat = swizzle_bytes * 8;
      static constexpr int subtile_cols = swizzle_bytes / sizeof(T);
      const int outer_idx = col / subtile_cols;
      const uint32_t addr = ptr +
          sizeof(T) *
              (outer_idx * ROWS * subtile_cols + row * subtile_cols +
               col % subtile_cols);
      const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
      return (addr ^ swizzle);
    } else {
      return ptr + sizeof(T) * (row * COLS + col);
    }
  }

  // Convenience functions to edit elements going through the swizzle.
  __device__ inline T& operator()(int row, int col) {
    return *ptr(data, row, col);
  }
  __device__ inline void store(float4& v, int row, int col) {
    *(reinterpret_cast<float4*>(ptr(data, row, col))) = v;
  }
  __device__ inline void store(float2& v, int row, int col) {
    *(reinterpret_cast<float2*>(ptr(data, row, col))) = v;
  }
  __device__ inline void store(float& v, int row, int col) {
    *(reinterpret_cast<float*>(ptr(data, row, col))) = v;
  }
  template <int N>
  __device__ inline void store(T (&v)[N], int row, int col) {
    if constexpr (sizeof(T) * N == 4) {
      store(*(reinterpret_cast<float*>(&v[0])), row, col);
    } else if constexpr (sizeof(T) * N == 8) {
      store(*(reinterpret_cast<float2*>(&v[0])), row, col);
    } else if constexpr (sizeof(T) * N == 16) {
      store(*(reinterpret_cast<float4*>(&v[0])), row, col);
    } else {
      MLX_UNROLL
      for (int i = 0; i < N; i++) {
        *ptr(data, row, col + i) = v[i];
      }
    }
  }
};

/**
 * Load the tile from global memory by loading 16 bytes at a time and storing
 * them immediately.
 *
 * Can also be used as a fallback for architectures before sm_80.
 */
template <int NUM_WARPS, typename T, typename Tile>
__device__ inline void load(Tile& tile, const T* x, int N) {
  constexpr int NUM_THREADS = NUM_WARPS * 32;
  constexpr int ELEMENTS_PER_LOAD = sizeof(float4) / sizeof(T);
  constexpr int NUM_LOADS = Tile::NUMEL / ELEMENTS_PER_LOAD;
  constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
  constexpr int NUM_LOADS_PER_ROW = Tile::COLS / ELEMENTS_PER_LOAD;
  constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;

  const int row = threadIdx.x / NUM_LOADS_PER_ROW;
  const int col = threadIdx.x % NUM_LOADS_PER_ROW;

  x += row * N + col * ELEMENTS_PER_LOAD;

  MLX_UNROLL
  for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
    float4 tmp;
    tmp = *(reinterpret_cast<const float4*>(&x[i * STEP_ROWS * N]));
    tile.store(tmp, row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD);
  }
}

/**
 * The asynchronous equivalent of load.
 *
 * Loads the tile from global memory by submitting a bunch of async copy
 * instructions. The copy won't start until commit is called and we don't have
 * a guarantee it will finish until wait is called.
 *
 * It should be used as follows
 *
 *    load(...)
 *    load(...)
 *    cp_async_commit()
 *    do_other_stuff()
 *    cp_async_wait_all()
 *    do_stuff_with_shmem()
 */
template <int NUM_WARPS, typename T, typename Tile>
__device__ inline void
load_async(Tile& tile, uint32_t base_address, const T* x, int N) {
  constexpr int NUM_THREADS = NUM_WARPS * 32;
  constexpr int ELEMENTS_PER_LOAD = sizeof(float4) / sizeof(T);
  constexpr int NUM_LOADS = Tile::NUMEL / ELEMENTS_PER_LOAD;
  constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
  constexpr int NUM_LOADS_PER_ROW = Tile::COLS / ELEMENTS_PER_LOAD;
  constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;

  const int row = threadIdx.x / NUM_LOADS_PER_ROW;
  const int col = threadIdx.x % NUM_LOADS_PER_ROW;

  x += row * N + col * ELEMENTS_PER_LOAD;

  MLX_UNROLL
  for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
    cp_async<16>(
        tile.loc(base_address, row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD),
        x + i * STEP_ROWS * N);
  }
}

/**
 * Same as load_async but checks if we can load the row.
 *
 * NOTE: It should be changed to use a predicated cp async instead.
 */
template <int NUM_WARPS, typename T, typename Tile>
__device__ inline void load_async_safe(
    Tile& tile,
    uint32_t base_address,
    const T* x,
    int N,
    int max_rows) {
  constexpr int NUM_THREADS = NUM_WARPS * 32;
  constexpr int ELEMENTS_PER_LOAD = sizeof(float4) / sizeof(T);
  constexpr int NUM_LOADS = Tile::NUMEL / ELEMENTS_PER_LOAD;
  constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
  constexpr int NUM_LOADS_PER_ROW = Tile::COLS / ELEMENTS_PER_LOAD;
  constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;

  const int row = threadIdx.x / NUM_LOADS_PER_ROW;
  const int col = threadIdx.x % NUM_LOADS_PER_ROW;

  x += row * N + col * ELEMENTS_PER_LOAD;

  MLX_UNROLL
  for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
    if (row + i * STEP_ROWS < max_rows) {
      cp_async<16>(
          tile.loc(base_address, row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD),
          x + i * STEP_ROWS * N);
    } else {
      float4 tmp = {0, 0, 0, 0};
      tile.store(tmp, row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD);
    }
  }
}

} // namespace mlx::core::cu
