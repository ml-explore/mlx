// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/dtype_utils.h"

namespace mlx::core {

namespace cu {

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

template <typename T>
struct Tile16x16 {
  using T2 = Vector2_t<T>;

  T2 values[4];

  __device__ inline void clear() {
    for (int i = 0; i < 4; i++) {
      values[i] = static_cast<T2>(0);
    }
  }

  __device__ inline void load(uint32_t src_address) {
    if constexpr (
        std::is_same_v<T2, __nv_bfloat162> || std::is_same_v<T2, __half2>) {
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(*(uint32_t*)&(values[0])),
            "=r"(*(uint32_t*)&(values[1])),
            "=r"(*(uint32_t*)&(values[2])),
            "=r"(*(uint32_t*)&(values[3]))
          : "r"(src_address));
    }
  }

  __device__ inline void store(uint32_t dst_address) {
    if constexpr (
        std::is_same_v<T2, __nv_bfloat162> || std::is_same_v<T2, __half2>) {
      asm volatile(
          "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(*(uint32_t*)&(values[0])),
            "=r"(*(uint32_t*)&(values[1])),
            "=r"(*(uint32_t*)&(values[2])),
            "=r"(*(uint32_t*)&(values[3]))
          : "r"(dst_address));
    } else {
      const int laneid = threadIdx.x % 32;
      const int row = laneid / 4;
      const int col = laneid % 4;

      const uint32_t a = dst_address + ((row + 0) * 8 + col + 0) * sizeof(T2);
      const uint32_t b = dst_address + ((row + 0) * 8 + col + 4) * sizeof(T2);
      const uint32_t c = dst_address + ((row + 8) * 8 + col + 0) * sizeof(T2);
      const uint32_t d = dst_address + ((row + 8) * 8 + col + 4) * sizeof(T2);
      if constexpr (sizeof(T2) == 4) {
        asm volatile("st.shared.b32 [%1], %0;\n"
                     :
                     : "r"(*(uint32_t*)&(values[0])), "r"(a));
        asm volatile("st.shared.b32 [%1], %0;\n"
                     :
                     : "r"(*(uint32_t*)&(values[2])), "r"(b));
        asm volatile("st.shared.b32 [%1], %0;\n"
                     :
                     : "r"(*(uint32_t*)&(values[1])), "r"(c));
        asm volatile("st.shared.b32 [%1], %0;\n"
                     :
                     : "r"(*(uint32_t*)&(values[3])), "r"(d));
      } else if constexpr (sizeof(T2) == 8) {
        asm volatile("st.shared.b64 [%1], %0;\n"
                     :
                     : "r"(*(uint64_t*)&(values[0])), "r"(a));
        asm volatile("st.shared.b64 [%1], %0;\n"
                     :
                     : "r"(*(uint64_t*)&(values[2])), "r"(b));
        asm volatile("st.shared.b64 [%1], %0;\n"
                     :
                     : "r"(*(uint64_t*)&(values[1])), "r"(c));
        asm volatile("st.shared.b64 [%1], %0;\n"
                     :
                     : "r"(*(uint64_t*)&(values[3])), "r"(d));
      } else if constexpr (sizeof(T2) == 16) {
        asm volatile("st.shared.b128 [%1], %0;\n"
                     :
                     : "r"(*(__int128*)&(values[0])), "r"(a));
        asm volatile("st.shared.b128 [%1], %0;\n"
                     :
                     : "r"(*(__int128*)&(values[2])), "r"(b));
        asm volatile("st.shared.b128 [%1], %0;\n"
                     :
                     : "r"(*(__int128*)&(values[1])), "r"(c));
        asm volatile("st.shared.b128 [%1], %0;\n"
                     :
                     : "r"(*(__int128*)&(values[3])), "r"(d));
      }
    }
  }

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
};

template <typename T, int ROWS, int COLS>
struct __align__(16) SharedTile {
  static constexpr int TILES_R = ROWS / 16;
  static constexpr int TILES_C = COLS / 16;
  static constexpr int NUM_ELEMENTS = ROWS * COLS;

  static constexpr int swizzle_bytes =
      (sizeof(T) == 2 ? (TILES_C % 4 == 0 ? 128 : (TILES_C % 2 == 0 ? 64 : 32))
                      : (sizeof(T) == 4 ? (TILES_C % 2 == 0 ? 128 : 64) : 0));

  T data[ROWS * COLS];

  __device__ static inline T* idx(T* ptr, int2 coord) {
    if constexpr (swizzle_bytes > 0) {
      int r = coord.x, c = coord.y;
      static constexpr int swizzle_repeat = swizzle_bytes * 8;
      static constexpr int subtile_cols = swizzle_bytes / sizeof(T);
      const int outer_idx = c / subtile_cols;
      const uint64_t addr =
          (uint64_t)(&ptr
                         [outer_idx * ROWS * subtile_cols + r * subtile_cols +
                          c % subtile_cols]);
      const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
      return (T*)(addr ^ swizzle);
    } else {
      return ptr + coord.y * COLS + coord.x;
    }
  }

  __device__ static inline uint32_t idx(uint32_t ptr, int2 coord) {
    if constexpr (swizzle_bytes > 0) {
      int r = coord.x, c = coord.y;
      static constexpr int swizzle_repeat = swizzle_bytes * 8;
      static constexpr int subtile_cols = swizzle_bytes / sizeof(T);
      const int outer_idx = c / subtile_cols;
      const uint32_t addr = ptr +
          sizeof(T) *
              (outer_idx * ROWS * subtile_cols + r * subtile_cols +
               c % subtile_cols);
      const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
      return (addr ^ swizzle);
    } else {
      return ptr + sizeof(T) * (coord.y * COLS + coord.x);
    }
  }

  __device__ inline T& operator[](int2 coord) {
    return *idx(&data[0], coord);
  }

  __device__ inline void store(float4& v, int2 coord) {
    *(reinterpret_cast<float4*>(idx(data, coord))) = v;
  }

  __device__ inline void store(float2& v, int2 coord) {
    *(reinterpret_cast<float2*>(idx(data, coord))) = v;
  }

  __device__ inline void store(float& v, int2 coord) {
    *(reinterpret_cast<float*>(idx(data, coord))) = v;
  }

  template <int N>
  __device__ inline void store(T (&v)[N], int2 coord) {
    if constexpr (sizeof(T) * N == 4) {
      store(*(reinterpret_cast<float*>(&v[0])), coord);
    } else if constexpr (sizeof(T) * N == 8) {
      store(*(reinterpret_cast<float2*>(&v[0])), coord);
    } else if constexpr (sizeof(T) * N == 16) {
      store(*(reinterpret_cast<float4*>(&v[0])), coord);
    } else {
#pragma unroll
      for (int i = 0; i < N; i++) {
        *idx(data, {coord.x, coord.y + i}) = v[i];
      }
    }
  }

  template <int NUM_WARPS>
  __device__ inline void load(const T* x, int N) {
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int ELEMENTS_PER_LOAD = sizeof(float4) / sizeof(T);
    constexpr int NUM_LOADS = NUM_ELEMENTS / ELEMENTS_PER_LOAD;
    constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
    constexpr int NUM_LOADS_PER_ROW = COLS / ELEMENTS_PER_LOAD;
    constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;

    const int row = threadIdx.x / NUM_LOADS_PER_ROW;
    const int col = threadIdx.x % NUM_LOADS_PER_ROW;

    x += row * N + col * ELEMENTS_PER_LOAD;

#pragma unroll
    for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
      float4 tmp;
      tmp = *(reinterpret_cast<const float4*>(&x[i * STEP_ROWS * N]));
      store(tmp, {row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD});
    }
  }

  template <int NUM_WARPS, int group_size, int bits>
  __device__ inline void
  load_quantized(const uint8_t* x, const T* scales, const T* biases, int N) {
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int ELEMENTS_PER_LOAD =
        sizeof(uint32_t) * get_pack_factor<bits>();
    constexpr int NUM_LOADS = NUM_ELEMENTS / ELEMENTS_PER_LOAD;
    constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
    constexpr int NUM_LOADS_PER_ROW = COLS / ELEMENTS_PER_LOAD;
    constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;
    constexpr int MASK = (1 << bits) - 1;

    const int row = threadIdx.x / NUM_LOADS_PER_ROW;
    const int col = threadIdx.x % NUM_LOADS_PER_ROW;

    const int Nx = N / get_pack_factor<bits>();
    const int Ng = N / group_size;

    x += row * Nx + col * (ELEMENTS_PER_LOAD / get_pack_factor<bits>());
    scales += row * Ng + col * ELEMENTS_PER_LOAD / group_size;
    biases += row * Ng + col * ELEMENTS_PER_LOAD / group_size;

#pragma unroll
    for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
      T vs[ELEMENTS_PER_LOAD];
      uint32_t w = *reinterpret_cast<const uint32_t*>(x + i * STEP_ROWS * Nx);
      T s = scales[i * STEP_ROWS * Ng];
      T b = biases[i * STEP_ROWS * Ng];
#pragma unroll
      for (int j = 0; j < ELEMENTS_PER_LOAD; j++) {
        vs[j] = static_cast<T>((w >> (j * bits)) & MASK) * s + b;
      }
      store(vs, {row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD});
    }
  }
};

template <typename TileAccum, typename Tile>
__device__ inline void mma(TileAccum& C, Tile& A, Tile& B) {}

__device__ inline void mma(
    Tile16x16<float>& C,
    Tile16x16<__nv_bfloat16>& A,
    Tile16x16<__nv_bfloat16>& B) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};"

      // D matrix
      : "+f"(C.values[0].x),
        "+f"(C.values[0].y),
        "+f"(C.values[1].x),
        "+f"(C.values[1].y)

      // A matrix
      : "r"(*(uint32_t*)(&A.values[0])),
        "r"(*(uint32_t*)(&A.values[1])),
        "r"(*(uint32_t*)(&A.values[2])),
        "r"(*(uint32_t*)(&A.values[3])),

        // B matrix
        "r"(*(uint32_t*)(&B.values[0])),
        "r"(*(uint32_t*)(&B.values[2])),

        // C matrix
        "f"(C.values[0].x),
        "f"(C.values[0].y),
        "f"(C.values[1].x),
        "f"(C.values[1].y));
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};"

      // D matrix
      : "+f"(C.values[2].x),
        "+f"(C.values[2].y),
        "+f"(C.values[3].x),
        "+f"(C.values[3].y)

      // A matrix
      : "r"(*(uint32_t*)(&A.values[0])),
        "r"(*(uint32_t*)(&A.values[1])),
        "r"(*(uint32_t*)(&A.values[2])),
        "r"(*(uint32_t*)(&A.values[3])),

        // B matrix
        "r"(*(uint32_t*)(&B.values[1])),
        "r"(*(uint32_t*)(&B.values[3])),

        // C matrix
        "f"(C.values[2].x),
        "f"(C.values[2].y),
        "f"(C.values[3].x),
        "f"(C.values[3].y));
}

template <typename T, int BM, int BN, int BK, int group_size, int bits>
__global__ void qmm(
    const T* x,
    const uint8_t* w,
    const T* scales,
    const T* biases,
    T* y,
    int M,
    int N,
    int K) {
  constexpr int NUM_WARPS = 4;
  constexpr int WARP_M = (BM / 16) / (NUM_WARPS / 2);
  constexpr int WARP_N = (BN / 16) / (NUM_WARPS / 2);
  constexpr int WARP_K = BK / 16;
  constexpr int WARP_STEP_M = WARP_M * 16;
  constexpr int WARP_STEP_N = WARP_N * 16;

  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int offset_m = (warpid / 2) * WARP_STEP_M;
  const int offset_n = (warpid % 2) * WARP_STEP_N;

  __shared__ SharedTile<T, BM, BK> xs;
  __shared__ SharedTile<T, BN, BK> ws;

  Tile16x16<float> C[WARP_M * WARP_N];
  Tile16x16<T> A[WARP_M];
  Tile16x16<T> B[WARP_N];

  x += blockIdx.y * BM * K;
  w += blockIdx.x * BN * K / get_pack_factor<bits>();
  scales += blockIdx.x * BN * K / group_size;
  biases += blockIdx.x * BN * K / group_size;
  y += blockIdx.y * BM * N + blockIdx.x * BN;

#pragma unroll
  for (int i = 0; i < WARP_M * WARP_N; i++) {
    C[i].clear();
  }

  uint32_t base_addr_xs = __cvta_generic_to_shared(&xs.data[0]);
  uint32_t base_addr_ws = __cvta_generic_to_shared(&ws.data[0]);

  for (int k_block = 0; k_block < K; k_block += BK) {
    xs.load<NUM_WARPS>(x + k_block, K);
    ws.load_quantized<NUM_WARPS, group_size, bits>(
        w + k_block / get_pack_factor<bits>(),
        scales + k_block / group_size,
        biases + k_block / group_size,
        K);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < WARP_K; k++) {
#pragma unroll
      for (int i = 0; i < WARP_M; i++) {
        A[i].load(xs.idx(
            base_addr_xs,
            {offset_m + i * 16 + laneid % 16, k * 16 + laneid / 16 * 8}));
      }
#pragma unroll
      for (int i = 0; i < WARP_N; i++) {
        B[i].load(ws.idx(
            base_addr_ws,
            {offset_n + i * 16 + laneid % 16, k * 16 + laneid / 16 * 8}));
      }

#pragma unroll
      for (int i = 0; i < WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N; j++) {
          mma(C[i * WARP_N + j], A[i], B[j]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < WARP_M; i++) {
#pragma unroll
    for (int j = 0; j < WARP_N; j++) {
      C[i * WARP_N + j].store_global(
          y + (offset_m + i * 16) * N + offset_n + j * 16, N);
    }
  }
}

} // namespace cu

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    bool transpose_,
    int group_size_,
    int bits_,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc,
    const Stream& s) {
  dispatch_float_types(x.dtype(), "qmm", [&](auto type_tag) {
    // dispatch_groups(group_size_, [&](auto group_size) {
    // dispatch_bits(bits_, [&](auto bits) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    auto kernel = cu::qmm<DataType, BM, BN, BK, 64, 4>;

    dim3 grid(N / BN, M / BM);

    enc.add_kernel_node(
        kernel,
        grid,
        128,
        x.data<DataType>(),
        w.data<uint8_t>(),
        scales.data<DataType>(),
        biases.data<DataType>(),
        out.data<DataType>(),
        M,
        N,
        K);
    //});
    //});
  });
}

} // namespace mlx::core
