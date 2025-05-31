// Updated from MLX commit has f70764a

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// ========================================== Generic vector types

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T>
struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B>
inline Acc mul(A a, B b);

template <typename T>
inline float sum(T v);

template <typename T>
inline float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// FP32 vector data types.
struct Float8_ {
  float4 x;
  float4 y;
};

template <>
struct Vec<float, 1> {
  using Type = float;
};
template <>
struct Vec<float, 2> {
  using Type = float2;
};
template <>
struct Vec<float, 4> {
  using Type = float4;
};
template <>
struct Vec<float, 8> {
  using Type = Float8_;
};

template <>
struct FloatVec<float> {
  using Type = float;
};
template <>
struct FloatVec<float2> {
  using Type = float2;
};
template <>
struct FloatVec<float4> {
  using Type = float4;
};
template <>
struct FloatVec<Float8_> {
  using Type = Float8_;
};

template <>
inline float mul(float a, float b) {
  return a * b;
}

template <>
inline float2 mul(float2 a, float2 b) {
  return a * b;
}

template <>
inline float4 mul(float4 a, float4 b) {
  return a * b;
}

template <>
inline Float8_ mul(Float8_ a, Float8_ b) {
  Float8_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline float sum(float a) {
  return a;
}

template <>
inline float sum(float2 a) {
  return a.x + a.y;
}

template <>
inline float sum(float4 a) {
  return a.x + a.y + a.z + a.w;
}

template <>
inline float sum(Float8_ a) {
  return sum(a.x) + sum(a.y);
}

inline Float8_ fma(Float8_ a, Float8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread float& dst, float src) {
  dst = src;
}
inline void from_float(thread float2& dst, float2 src) {
  dst = src;
}
inline void from_float(thread float4& dst, float4 src) {
  dst = src;
}
inline void from_float(thread Float8_& dst, Float8_ src) {
  dst = src;
}

// BF16 vector data types.
// #if defined(__HAVE_BFLOAT__)

// struct Bfloat8_ {
//   bfloat4 x;
//   bfloat4 y;
// };

// template<>
// struct Vec<bfloat, 1> {
//   using Type = bfloat;
// };
// template<>
// struct Vec<bfloat, 2> {
//   using Type = bfloat2;
// };
// template<>
// struct Vec<bfloat, 4> {
//   using Type = bfloat4;
// };
// template<>
// struct Vec<bfloat, 8> {
//   using Type = Bfloat8_;
// };

// template<>
// struct FloatVec<bfloat> {
//   using Type = float;
// };
// template<>
// struct FloatVec<bfloat2> {
//   using Type = float2;
// };
// template<>
// struct FloatVec<bfloat4> {
//   using Type = float4;
// };
// template<>
// struct FloatVec<Bfloat8_> {
//   using Type = Float8_;
// };

// template<>
// inline float mul(bfloat a, bfloat b) {
//   return (float)a * (float)b;
// }
// template<>
// inline bfloat mul(bfloat a, bfloat b) {
//   return a*b;
// }

// template<>
// inline float2 mul(bfloat2 a, bfloat2 b) {
//   return (float2)a * (float2)b;
// }
// template<>
// inline bfloat2 mul(bfloat2 a, bfloat2 b) {
//   return a * b;
// }

// template<>
// inline float4 mul(bfloat4 a, bfloat4 b) {
//   return (float4)a * (float4)b;
// }
// template<>
// inline bfloat4 mul(bfloat4 a, bfloat4 b) {
//   return a * b;
// }

// template<>
// inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Float8_ c;
//   c.x = mul<float4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<float4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }
// template<>
// inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Bfloat8_ c;
//   c.x = mul<bfloat4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<bfloat4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }

// template<>
// inline float sum(bfloat a) {
//   return (float)a;
// }

// template<>
// inline float sum(bfloat2 a) {
//   return (float)a.x + (float)a.y;
// }

// template<>
// inline float sum(bfloat4 a) {
//   return sum(a.x) + sum(a.y);
// }

// template<>
// inline float sum(Bfloat8_ a) {
//   return sum(a.x) + sum(a.y);
// }

// inline float fma(bfloat a, bfloat b, float c) {
//   return (float)a * (float)b + c;
// }

// inline float2 fma(bfloat2 a, bfloat2 b, float2 c) {
//   return (float2)a * (float2)b + c;
// }

// inline float4 fma(bfloat4 a, bfloat4 b, float4 c) {
//   return (float4)a * (float4)b + c;
// }

// inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
//   Float8_ res;
//   res.x = fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = fma((float4)a.y, (float4)b.y, (float4)c.y);
//   return res;
// }
// inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
//   Bfloat8_ res;
//   res.x = (bfloat4)fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = (bfloat4)fma((float4)a.y, (float4)b.x, (float4)c.y);
//   return c;
// }

// inline void from_float(thread bfloat& dst, float src) {
//   dst = static_cast<bfloat>(src);
// }
// inline void from_float(thread bfloat2& dst, float2 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
// }
// inline void from_float(thread bfloat4& dst, float4 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
//   dst.z = static_cast<bfloat>(src.z);
//   dst.w = static_cast<bfloat>(src.w);
// }
// inline void from_float(thread Bfloat8_& dst, Float8_ src) {
//   bfloat4 x;
//   bfloat4 y;
//   from_float(x, src.x);
//   from_float(y, src.y);
//   dst.x = x;
//   dst.y = y;
// }

// #else

struct Bfloat2_ {
  bfloat16_t x;
  bfloat16_t y;
};

struct Bfloat4_ {
  Bfloat2_ x;
  Bfloat2_ y;
};

struct Bfloat8_ {
  Bfloat4_ x;
  Bfloat4_ y;
};

template <>
struct Vec<bfloat16_t, 1> {
  using Type = bfloat16_t;
};
template <>
struct Vec<bfloat16_t, 2> {
  using Type = Bfloat2_;
};
template <>
struct Vec<bfloat16_t, 4> {
  using Type = Bfloat4_;
};
template <>
struct Vec<bfloat16_t, 8> {
  using Type = Bfloat8_;
};

template <>
struct FloatVec<bfloat16_t> {
  using Type = float;
};
template <>
struct FloatVec<Bfloat2_> {
  using Type = float2;
};
template <>
struct FloatVec<Bfloat4_> {
  using Type = float4;
};
template <>
struct FloatVec<Bfloat8_> {
  using Type = Float8_;
};

template <>
inline float mul(bfloat16_t a, bfloat16_t b) {
  return (float)a * (float)b;
}
template <>
inline bfloat16_t mul(bfloat16_t a, bfloat16_t b) {
  return a * b;
}

template <>
inline float2 mul(Bfloat2_ a, Bfloat2_ b) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f;
}
template <>
inline Bfloat2_ mul(Bfloat2_ a, Bfloat2_ b) {
  Bfloat2_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline float4 mul(Bfloat4_ a, Bfloat4_ b) {
  float2 x = mul<float2, Bfloat2_, Bfloat2_>(a.x, b.x);
  float2 y = mul<float2, Bfloat2_, Bfloat2_>(a.y, b.y);
  float4 c;
  c.x = x.x;
  c.y = x.y;
  c.z = y.x;
  c.w = y.y;
  return c;
}
template <>
inline Bfloat4_ mul(Bfloat4_ a, Bfloat4_ b) {
  Bfloat4_ c;
  c.x = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.x, b.x);
  c.y = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.y, b.y);
  return c;
}

template <>
inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Float8_ c;
  c.x = mul<float4, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<float4, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}
template <>
inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Bfloat8_ c;
  c.x = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}

template <>
inline float sum(bfloat16_t a) {
  return (float)a;
}

template <>
inline float sum(Bfloat2_ a) {
  return (float)a.x + (float)a.y;
}

template <>
inline float sum(Bfloat4_ a) {
  return sum(a.x) + sum(a.y);
}

template <>
inline float sum(Bfloat8_ a) {
  return sum(a.x) + sum(a.y);
}

inline float fma(bfloat16_t a, bfloat16_t b, float c) {
  return (float)a * (float)b + c;
}
inline bfloat16_t fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
  return a * b + c;
}

inline float2 fma(Bfloat2_ a, Bfloat2_ b, float2 c) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f + c;
}
inline Bfloat2_ fma(Bfloat2_ a, Bfloat2_ b, Bfloat2_ c) {
  Bfloat2_ res;
  res.x = a.x * b.x + c.x;
  res.y = a.y * b.y + c.y;
  return res;
}

inline float4 fma(Bfloat4_ a, Bfloat4_ b, float4 c) {
  float4 res;
  res.x = fma(a.x.x, b.x.x, c.x);
  res.y = fma(a.x.y, b.x.y, c.y);
  res.z = fma(a.y.x, b.y.x, c.z);
  res.w = fma(a.y.y, b.y.y, c.w);
  return res;
}
inline Bfloat4_ fma(Bfloat4_ a, Bfloat4_ b, Bfloat4_ c) {
  Bfloat4_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
  Bfloat8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread bfloat16_t& dst, float src) {
  dst = static_cast<bfloat16_t>(src);
}
inline void from_float(thread Bfloat2_& dst, float2 src) {
  dst.x = static_cast<bfloat16_t>(src.x);
  dst.y = static_cast<bfloat16_t>(src.y);
}
inline void from_float(thread Bfloat4_& dst, float4 src) {
  dst.x.x = static_cast<bfloat16_t>(src.x);
  dst.x.y = static_cast<bfloat16_t>(src.y);
  dst.y.x = static_cast<bfloat16_t>(src.z);
  dst.y.y = static_cast<bfloat16_t>(src.w);
}
inline void from_float(thread Bfloat8_& dst, Float8_ src) {
  Bfloat4_ x;
  Bfloat4_ y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// #endif

// FP16 vector data types.
struct Half8_ {
  half4 x;
  half4 y;
};

template <>
struct Vec<half, 1> {
  using Type = half;
};
template <>
struct Vec<half, 2> {
  using Type = half2;
};
template <>
struct Vec<half, 4> {
  using Type = half4;
};
template <>
struct Vec<half, 8> {
  using Type = Half8_;
};

template <>
struct FloatVec<half> {
  using Type = float;
};
template <>
struct FloatVec<half2> {
  using Type = float2;
};
template <>
struct FloatVec<half4> {
  using Type = float4;
};
template <>
struct FloatVec<Half8_> {
  using Type = Float8_;
};

template <>
inline float mul(half a, half b) {
  return (float)a * (float)b;
}
template <>
inline half mul(half a, half b) {
  return a * b;
}

template <>
inline float2 mul(half2 a, half2 b) {
  return (float2)a * (float2)b;
}
template <>
inline half2 mul(half2 a, half2 b) {
  return a * b;
}

template <>
inline float4 mul(half4 a, half4 b) {
  return (float4)a * (float4)b;
}
template <>
inline half4 mul(half4 a, half4 b) {
  return a * b;
}

template <>
inline Float8_ mul(Half8_ a, Half8_ b) {
  float4 x = mul<float4, half4, half4>(a.x, b.x);
  float4 y = mul<float4, half4, half4>(a.y, b.y);
  Float8_ c;
  c.x = x;
  c.y = y;
  return c;
}
template <>
inline Half8_ mul(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = mul<half4, half4, half4>(a.x, b.x);
  c.y = mul<half4, half4, half4>(a.y, b.y);
  return c;
}

template <>
inline float sum(half a) {
  return (float)a;
}

template <>
inline float sum(half2 a) {
  return (float)a.x + (float)a.y;
}

template <>
inline float sum(half4 a) {
  return a.x + a.y + a.z + a.w;
}

template <>
inline float sum(Half8_ a) {
  return sum(a.x) + sum(a.y);
}

inline float fma(half a, half b, float c) {
  return (float)a * (float)b + c;
}

inline float2 fma(half2 a, half2 b, float2 c) {
  return (float2)a * (float2)b + c;
}

inline float4 fma(half4 a, half4 b, float4 c) {
  return (float4)a * (float4)b + c;
}

inline Float8_ fma(Half8_ a, Half8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Half8_ fma(Half8_ a, Half8_ b, Half8_ c) {
  Half8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread half& dst, float src) {
  dst = static_cast<half>(src);
}
inline void from_float(thread half2& dst, float2 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
}
inline void from_float(thread half4& dst, float4 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
  dst.z = static_cast<half>(src.z);
  dst.w = static_cast<half>(src.w);
}
inline void from_float(thread Half8_& dst, Float8_ src) {
  half4 x;
  half4 y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// ========================================== Dot product utilities

// TODO(EricLBuehler): optimize with vectorization
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  using A_vec = typename FloatVec<Vec>::Type;
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += simd_shuffle_xor(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(
      const threadgroup Vec (&q)[N],
      const thread Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// ========================================== Block sum utility

// Utility function for attention softmax.
template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(
    threadgroup float* red_smem,
    float sum,
    uint simd_tid,
    uint simd_lid) {
  // Compute the sum per simdgroup.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Simd leaders store the data to shared memory.
  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }

  // Make sure the data is in shared memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The warps compute the final sums.
  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  }

  // Parallel reduction inside the simd group.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Broadcast to other threads.
  return simd_shuffle(sum, 0);
}

// ========================================== Paged Attention kernel

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

constant bool use_partitioning [[function_constant(10)]];
constant bool use_alibi [[function_constant(20)]];

template <
    typename T,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int NUM_SIMD_LANES,
    int PARTITION_SIZE = 0>
[[kernel]] void paged_attention(
    device float* exp_sums
    [[buffer(0), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device float* max_logits
    [[buffer(1), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device T* out
    [[buffer(2)]], // [num_seqs, num_heads, max_num_partitions, head_size]
    device const T* q [[buffer(3)]], // [num_seqs, num_heads, head_size]
    device const T* k_cache
    [[buffer(4)]], // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    device const T* v_cache
    [[buffer(5)]], // [num_blocks, num_kv_heads, head_size, block_size]
    const constant int& num_kv_heads [[buffer(6)]], // [num_heads]
    const constant float& scale [[buffer(7)]],
    const constant float& softcapping [[buffer(8)]],
    device const uint32_t* block_tables
    [[buffer(9)]], // [num_seqs, max_num_blocks_per_seq]
    device const uint32_t* context_lens [[buffer(10)]], // [num_seqs]
    const constant int& max_num_blocks_per_seq [[buffer(11)]],
    device const float* alibi_slopes
    [[buffer(12), function_constant(use_alibi)]], // [num_heads]
    const constant int& q_stride [[buffer(13)]],
    const constant int& kv_block_stride [[buffer(14)]],
    const constant int& kv_head_stride [[buffer(15)]],
    threadgroup char* shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int seq_idx = threadgroup_position_in_grid.y;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = thread_position_in_threadgroup.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const uint32_t context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = !use_alibi ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in the
  // group has 0, 4, 8, ... th vectors of the query, and the second thread has
  // 1, 5, 9, ... th vectors of the query, and so on.
  const device T* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  threadgroup Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const device Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Use fp32 on softmax logits for better accuracy
  threadgroup float* logits = reinterpret_cast<threadgroup float*>(shared_mem);
  // Workspace for reduction
  threadgroup float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(T);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const device uint32_t* block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE: The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by
    // large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the thread group size is 4, then the first thread in the
    // group has 0, 4, 8, ... th vectors of the key, and the second thread has
    // 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const device T* k_ptr = k_cache +
            physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const device K_vec*>(
            k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale *
          Qk_dot<T, THREAD_GROUP_SIZE>::dot(
                     q_vecs[thread_group_offset], k_vecs);

      // Apply softcapping
      if (softcapping != 1.0) {
        qk = precise::tanh(qk / softcapping) * softcapping;
      }

      // Add the ALiBi bias if slopes are given.
      qk +=
          (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE: It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : max(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = simd_shuffle(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(
      &red_smem[NUM_WARPS], exp_sum, simd_tid, simd_lid);

  // Compute softmax.
  const float inv_sum = divide(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0 && use_partitioning) {
    device float* max_logits_ptr = max_logits +
        seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    device float* exp_sums_ptr = exp_sums +
        seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(T), BLOCK_SIZE);
  using V_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = NUM_SIMD_LANES / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE: We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  T zero_value = 0;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE: The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by
    // large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    Float_L_vec logits_float_vec = *reinterpret_cast<threadgroup Float_L_vec*>(
        logits + token_idx - start_token_idx);
    from_float(logits_vec, logits_float_vec);

    const device T* v_ptr = v_cache + physical_block_number * kv_block_stride +
        kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        // NOTE: When v_vec contains the tokens that are out of the context,
        // we should explicitly zero out the values since they may contain NaNs.
        // See
        // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
        V_vec v_vec = *reinterpret_cast<const device V_vec*>(v_ptr + offset);
        if (block_idx == num_context_blocks - 1) {
          thread T* v_vec_ptr = reinterpret_cast<thread T*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] =
                token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += simd_shuffle_xor(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE: A barrier is required because the shared memory space for logits
  // is reused for the output.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Perform reduction across warps.
  threadgroup float* out_smem =
      reinterpret_cast<threadgroup float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      threadgroup float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lower warps update the output.
    if (warp_idx < mid) {
      const threadgroup float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final output.
  if (warp_idx == 0) {
    device T* out_ptr = out +
        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        *(out_ptr + row_idx) = T(accs[i]);
      }
    }
  }
}

template <
    typename T,
    int HEAD_SIZE,
    int NUM_THREADS,
    int NUM_SIMD_LANES,
    int PARTITION_SIZE = 0>
[[kernel]] void paged_attention_v2_reduce(
    device T* out [[buffer(0)]],
    const device float* exp_sums [[buffer(1)]],
    const device float* max_logits [[buffer(2)]],
    const device T* tmp_out [[buffer(3)]],
    device uint32_t* context_lens [[buffer(4)]],
    const constant int& max_num_partitions [[buffer(5)]],
    threadgroup char* shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int num_heads = threadgroups_per_grid.x;
  const int head_idx = threadgroup_position_in_grid.x;
  const int seq_idx = threadgroup_position_in_grid.y;
  const uint32_t context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    device T* out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const device T* tmp_out_ptr = tmp_out +
        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
         i += threads_per_threadgroup.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  // Workspace for reduction.
  threadgroup float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  threadgroup float* shared_max_logits =
      reinterpret_cast<threadgroup float*>(shared_mem);
  const device float* max_logits_ptr = max_logits +
      seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = max(max_logit, l);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = simd_shuffle(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  threadgroup float* shared_exp_sums = reinterpret_cast<threadgroup float*>(
      shared_mem + sizeof(float) * num_partitions);
  const device float* exp_sums_ptr = exp_sums +
      seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * exp(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(
      &red_smem[NUM_WARPS], global_exp_sum, simd_tid, simd_lid);
  const float inv_global_exp_sum = divide(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const device T* tmp_out_ptr = tmp_out +
      seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  device T* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
       i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
          inv_global_exp_sum;
    }
    out_ptr[i] = T(acc);
  }
}
