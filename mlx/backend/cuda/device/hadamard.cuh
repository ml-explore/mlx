// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/utils.cuh"

namespace mlx::core::cu {

__device__ __forceinline__ void hadamard_radix_m(float* x);

template <int N>
struct Pow2Log2 {
  static_assert(
      (N > 0) && ((N & (N - 1)) == 0),
      "N must be a positive power of two.");
  static constexpr int value = 1 + Pow2Log2<N / 2>::value;
};

template <>
struct Pow2Log2<1> {
  static constexpr int value = 0;
};

template <int R>
__device__ __forceinline__ void hadamard_radix_pow2(float* x) {
  constexpr int kLogR = Pow2Log2<R>::value;
  int h = 1;
#pragma unroll
  for (int s = 0; s < kLogR; ++s) {
#pragma unroll
    for (int i = 0; i < R / 2; ++i) {
      int k = i & (h - 1);
      int j = ((i - k) << 1) + k;
      float a = x[j];
      float b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    h <<= 1;
  }
}

template <typename T, int N, int max_radix, int read_width, int stride = 1>
__global__ void
hadamard_n(const T* in, T* out, float scale, long long num_transforms) {
  constexpr int kNumThreads = N / max_radix;
  constexpr int kLogN = Pow2Log2<N>::value;
  constexpr int kLogR = Pow2Log2<max_radix>::value;
  constexpr int kNumSteps = kLogN / kLogR;
  constexpr int kLogFinal = kLogN % kLogR;
  constexpr int kFinalRadix = 1 << kLogFinal;

  if (threadIdx.x >= kNumThreads) {
    return;
  }

  __shared__ T buf[N];
  int i = threadIdx.x;

  for (long long transform = blockIdx.x; transform < num_transforms;
       transform += gridDim.x) {
    long long base = (transform / stride) * static_cast<long long>(N) * stride +
        (transform % stride);

    if constexpr (stride == 1) {
#pragma unroll
      for (int j = 0; j < max_radix / read_width; ++j) {
        int index = j * read_width * kNumThreads + i * read_width;
#pragma unroll
        for (int r = 0; r < read_width; ++r) {
          buf[index + r] = in[base + index + r];
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < max_radix; ++j) {
        buf[j * kNumThreads + i] = in[base + (j * kNumThreads + i) * stride];
      }
    }
    __syncthreads();

    float x[max_radix];
    int h = 1;

#pragma unroll
    for (int s = 0; s < kNumSteps; ++s) {
      int k = i & (h - 1);
      int j = ((i - k) << kLogR) + k;

#pragma unroll
      for (int r = 0; r < max_radix; ++r) {
        x[r] = static_cast<float>(buf[j + h * r]);
      }

      hadamard_radix_pow2<max_radix>(x);

#pragma unroll
      for (int r = 0; r < max_radix; ++r) {
        buf[j + h * r] = static_cast<T>(x[r]);
      }

      h <<= kLogR;
      __syncthreads();
    }

    if constexpr (kFinalRadix > 1) {
#pragma unroll
      for (int t = 0; t < max_radix / kFinalRadix; ++t) {
        int index = i + t * kNumThreads;
        int k = index & (h - 1);
        int j = ((index - k) << kLogFinal) + k;
#pragma unroll
        for (int r = 0; r < kFinalRadix; ++r) {
          x[r] = static_cast<float>(buf[j + h * r]);
        }

        hadamard_radix_pow2<kFinalRadix>(x);

#pragma unroll
        for (int r = 0; r < kFinalRadix; ++r) {
          buf[j + h * r] = static_cast<T>(x[r]);
        }
      }
      __syncthreads();
    }

    if constexpr (stride == 1) {
#pragma unroll
      for (int j = 0; j < max_radix / read_width; ++j) {
        int index = j * read_width * kNumThreads + i * read_width;
#pragma unroll
        for (int r = 0; r < read_width; ++r) {
          float val = static_cast<float>(buf[index + r]);
          out[base + index + r] = static_cast<T>(val * scale);
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < max_radix; ++j) {
        out[base + (j * kNumThreads + i) * stride] = buf[j * kNumThreads + i];
      }
    }

    __syncthreads();
  }
}

template <typename T, int N, int M, int read_width>
__global__ void
hadamard_m(const T* in, T* out, float scale, long long num_tasks) {
  constexpr int kTasksPerBatch = N / read_width;

  for (long long task = blockIdx.x * blockDim.x + threadIdx.x; task < num_tasks;
       task += blockDim.x * gridDim.x) {
    long long i = task % kTasksPerBatch;
    long long batch = task / kTasksPerBatch;
    long long base = batch * static_cast<long long>(M) * N;

    float x[read_width][M];
#pragma unroll
    for (int c = 0; c < M; ++c) {
#pragma unroll
      for (int r = 0; r < read_width; ++r) {
        x[r][c] = static_cast<float>(in[base + c * N + i * read_width + r]);
      }
    }

#pragma unroll
    for (int r = 0; r < read_width; ++r) {
      hadamard_radix_m(x[r]);
    }

#pragma unroll
    for (int c = 0; c < M; ++c) {
#pragma unroll
      for (int r = 0; r < read_width; ++r) {
        out[base + c * N + i * read_width + r] =
            static_cast<T>(x[r][c] * scale);
      }
    }
  }
}

} // namespace mlx::core::cu
