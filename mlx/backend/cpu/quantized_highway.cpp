// Copyright © 2026 Apple Inc.

// Normally this file is compiled directly and Highway emits its runtime
// dispatch targets. Native MSVC builds compile this file once per target with
// MLX_HIGHWAY_MANUAL_TARGET and MLX_HIGHWAY_TARGET_SUFFIX so the same kernels
// are emitted as one manually suffixed specialization.

#include <atomic>
#include <stdexcept>

#if !defined(MLX_HIGHWAY_MANUAL_TARGET)
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/quantized_highway.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep
#endif

#include "hwy/highway.h"
#include "mlx/backend/cpu/highway_utils.h"
#include "mlx/backend/cpu/quantized_highway.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;
namespace hu = mlx::core::highway::HWY_NAMESPACE;
using hu::load_typed_as_f32;

// Highway's AVX3-family x86 targets use 512-bit vectors.
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_DL ||       \
    HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_SPR || \
    HWY_TARGET == HWY_AVX10_2
constexpr bool kHas512BitX86Vectors = true;
#else
constexpr bool kHas512BitX86Vectors = false;
#endif

template <class DU8>
hn::Vec<DU8> unpack_4bit_lanes(DU8 du8, const uint32_t* words) {
#if HWY_TARGET == HWY_AVX2
  const auto packed =
      hn::LoadDup128(du8, reinterpret_cast<const uint8_t*>(words));
#else
  const auto packed = hn::LoadN(
      du8, reinterpret_cast<const uint8_t*>(words), hn::Lanes(du8) / 2);
#endif
  const auto mask = hn::Set(du8, uint8_t{0x0F});
  const auto lo = hn::And(packed, mask);
  const auto hi = hn::And(hn::ShiftRight<4>(packed), mask);
#if HWY_MAX_BYTES == 16
  return hn::InterleaveLower(lo, hi);
#else
  return hn::ConcatLowerLower(
      du8, hn::InterleaveUpper(du8, lo, hi), hn::InterleaveLower(lo, hi));
#endif
}

template <int bits, class DU32>
hn::Vec<DU32> unpack_packed_words(DU32 du32, const uint32_t* words) {
  static_assert(bits == 4 || bits == 8);
  constexpr int pack_factor = 32 / bits;
  const auto lanes = hn::Lanes(du32);
  const auto lane_idx = hn::Iota(du32, uint32_t{0});
  const auto elem_idx =
      hn::And(lane_idx, hn::Set(du32, uint32_t{pack_factor - 1}));
  const auto shifts = hn::Mul(elem_idx, hn::Set(du32, uint32_t{bits}));
  const auto mask = hn::Set(du32, uint32_t{(1u << bits) - 1});

  if (lanes == pack_factor) {
    return hn::And(hn::Shr(hn::Set(du32, *words), shifts), mask);
  }

  const int words_per_vec = static_cast<int>(lanes) / pack_factor;
  const auto packed = hn::LoadN(du32, words, words_per_vec);
  auto word_idx = hn::Zero(du32);
  if constexpr (bits == 4) {
    word_idx = hn::ShiftRight<3>(lane_idx);
  } else {
    word_idx = hn::ShiftRight<2>(lane_idx);
  }
  const auto packed_for_lane =
      hn::TableLookupLanes(packed, hn::IndicesFromVec(du32, word_idx));
  return hn::And(hn::Shr(packed_for_lane, shifts), mask);
}

template <typename OutT, typename T, int bits, int group_size, int NC>
void qmm_t_int8_cols(
    OutT* HWY_RESTRICT result,
    const int8_t* HWY_RESTRICT x_q,
    const float* HWY_RESTRICT x_scales,
    const float* HWY_RESTRICT x_group_sums,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int n,
    int K) {
  static_assert(bits == 4 || bits == 8);

  constexpr int pack_factor = 32 / bits;
  const int groups_per_col = K / group_size;
  const int packs_in_group = group_size / pack_factor;
  const int packs_per_col = groups_per_col * packs_in_group;

  constexpr int lane_cap =
      (kHas512BitX86Vectors && bits == 8 && group_size >= 64) ? 64 : 32;
  const hn::CappedTag<int8_t, lane_cap> di8;
  const hn::RebindToUnsigned<decltype(di8)> du8;
  const hn::Repartition<int16_t, decltype(di8)> di16;
  const hn::Repartition<int32_t, decltype(di8)> di32;
  const hn::Rebind<float, decltype(di32)> df;
  const size_t lanes = hn::Lanes(di8);
  const int words_per_vec = static_cast<int>(lanes) / pack_factor;
  const auto ones16 = hn::Set(di16, int16_t{1});

  hn::Vec<decltype(df)> accum_vec[NC];
  for (int c = 0; c < NC; ++c) {
    accum_vec[c] = hn::Zero(df);
  }
  float bias_accum[NC] = {};

  for (int g = 0; g < groups_per_col; ++g) {
    hn::Vec<decltype(di32)> dot_acc[NC];
    for (int c = 0; c < NC; ++c) {
      dot_acc[c] = hn::Zero(di32);
    }

    const int8_t* x_group = x_q + g * group_size;
    const uint32_t* w_group[NC];
    for (int c = 0; c < NC; ++c) {
      w_group[c] = w + (n + c) * packs_per_col + g * packs_in_group;
    }
    for (int elem = 0; elem < group_size; elem += static_cast<int>(lanes)) {
      const auto x_vec = hn::Load(di8, x_group + elem);
      HWY_UNROLL(4)
      for (int c = 0; c < NC; ++c) {
        hn::Vec<decltype(du8)> w_vec;
        if constexpr (bits == 4) {
          w_vec = unpack_4bit_lanes(du8, w_group[c]);
          w_group[c] += words_per_vec;
          const auto prod16 = hn::SatWidenMulPairwiseAdd(di16, w_vec, x_vec);
          auto unused = hn::Zero(di32);
          dot_acc[c] = hn::ReorderWidenMulAccumulate(
              di32, prod16, ones16, dot_acc[c], unused);
        } else {
          const auto* w_bytes = reinterpret_cast<const uint8_t*>(w_group[c]);
          w_group[c] += words_per_vec;
          w_vec = hn::LoadU(du8, w_bytes);
          dot_acc[c] =
              hn::SumOfMulQuadAccumulate(di32, w_vec, x_vec, dot_acc[c]);
        }
      }
    }

    const float xs = x_scales[g];
    const float xgs = x_group_sums[g];
    for (int c = 0; c < NC; ++c) {
      const size_t param_idx = static_cast<size_t>(n + c) * groups_per_col + g;
      const float scale_f = static_cast<float>(scales[param_idx]);
      const float bias_f = static_cast<float>(biases[param_idx]);
      accum_vec[c] = hn::MulAdd(
          hn::Set(df, scale_f * xs),
          hn::ConvertTo(df, dot_acc[c]),
          accum_vec[c]);
      bias_accum[c] += bias_f * xgs;
    }
  }

  for (int c = 0; c < NC; ++c) {
    result[c] =
        static_cast<OutT>(hn::ReduceSum(df, accum_vec[c]) + bias_accum[c]);
  }
}

template <int bits>
void dequant_row(
    const uint32_t* HWY_RESTRICT w_row,
    const float* HWY_RESTRICT scales_row,
    const float* HWY_RESTRICT biases_row,
    float* HWY_RESTRICT out,
    int group_size,
    int K) {
  static_assert(bits == 4 || bits == 8);
  constexpr int pack_factor = 32 / bits;
  constexpr int lane_cap = kHas512BitX86Vectors ? 16 : (bits == 4 ? 8 : 4);

  const hn::CappedTag<uint32_t, lane_cap> du;
  const hn::Rebind<float, decltype(du)> df;
  using VU = hn::Vec<decltype(du)>;
  using VF = hn::Vec<decltype(df)>;

  const int lanes = static_cast<int>(hn::Lanes(du));
  const VU bit_mask = hn::Set(du, (1u << bits) - 1);
  const VU bit_width = hn::Set(du, bits);

  int k = 0;
  const uint32_t* w_ptr = w_row;
  for (int g = 0; g < K / group_size; ++g) {
    const VF scale = hn::Set(df, scales_row[g]);
    const VF bias = hn::Set(df, biases_row[g]);

    if (lanes >= pack_factor) {
      const int words_per_vec = lanes / pack_factor;
      for (int j = 0; j < group_size; j += lanes) {
        const VU values = unpack_packed_words<bits>(du, w_ptr);
        w_ptr += words_per_vec;
        hn::StoreU(
            hn::MulAdd(hn::ConvertTo(df, values), scale, bias), df, out + k);
        k += lanes;
      }
    } else {
      for (int j = 0; j < group_size; j += pack_factor) {
        const VU packed = hn::Set(du, *w_ptr++);
        for (int elem = 0; elem < pack_factor; elem += lanes) {
          const VU shifts = hn::Mul(hn::Iota(du, elem), bit_width);
          const VU values = hn::And(hn::Shr(packed, shifts), bit_mask);
          hn::StoreU(
              hn::MulAdd(hn::ConvertTo(df, values), scale, bias),
              df,
              out + k + elem);
        }
        k += pack_factor;
      }
    }
  }
}

template <typename T>
void QuantizeActivationInt8Typed(
    const T* HWY_RESTRICT x,
    int K,
    int group_size,
    int8_t* HWY_RESTRICT x_q,
    float* HWY_RESTRICT x_scales,
    float* HWY_RESTRICT x_group_sums) {
  const hn::ScalableTag<float> df;
  const hn::RebindToSigned<decltype(df)> di32;
  const hn::Repartition<int16_t, decltype(di32)> di16;
  const hn::Repartition<int8_t, decltype(di32)> di8;
  using VF = hn::Vec<decltype(df)>;

  const size_t lanes = hn::Lanes(df);
  alignas(64) int32_t q_tmp[16];
  const int groups = K / group_size;
  for (int g = 0; g < groups; ++g) {
    const size_t group_offset = static_cast<size_t>(g) * group_size;
    VF sum = hn::Zero(df);
    VF amax = hn::Zero(df);

    for (int e = 0; e < group_size; e += static_cast<int>(lanes)) {
      const VF v = load_typed_as_f32(df, x, group_offset + e);
      sum = hn::Add(sum, v);
      amax = hn::Max(amax, hn::Abs(v));
    }

    x_group_sums[g] = hn::ReduceSum(df, sum);
    const float max_value = hn::ReduceMax(df, amax);
    const float inv_scale = max_value > 0.0f ? 127.0f / max_value : 0.0f;
    x_scales[g] = max_value / 127.0f;
    const VF inv = hn::Set(df, inv_scale);

    if (lanes == 16 && group_size % 64 == 0) {
      for (int e = 0; e < group_size; e += 64) {
        const auto q0 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e), inv));
        const auto q1 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 16), inv));
        const auto q2 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 32), inv));
        const auto q3 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 48), inv));
        hn::Store(
            hn::OrderedDemote2To(
                di8,
                hn::OrderedDemote2To(di16, q0, q1),
                hn::OrderedDemote2To(di16, q2, q3)),
            di8,
            x_q + group_offset + e);
      }
    } else if (lanes == 16) {
      const hn::Half<decltype(di8)> di8_half;
      for (int e = 0; e < group_size; e += 32) {
        const auto q0 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e), inv));
        const auto q1 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 16), inv));
        hn::Store(
            hn::DemoteTo(di8_half, hn::OrderedDemote2To(di16, q0, q1)),
            di8_half,
            x_q + group_offset + e);
      }
    } else if (lanes == 8) {
      for (int e = 0; e < group_size; e += 32) {
        const auto q0 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e), inv));
        const auto q1 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 8), inv));
        const auto q2 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 16), inv));
        const auto q3 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 24), inv));
        hn::Store(
            hn::OrderedDemote2To(
                di8,
                hn::OrderedDemote2To(di16, q0, q1),
                hn::OrderedDemote2To(di16, q2, q3)),
            di8,
            x_q + group_offset + e);
      }
    } else {
      for (int e = 0; e < group_size; e += static_cast<int>(lanes)) {
        const VF v = load_typed_as_f32(df, x, group_offset + e);
        const auto q = hn::NearestInt(hn::Mul(v, inv));
        hn::StoreU(q, di32, q_tmp);
        for (size_t lane = 0; lane < lanes; ++lane) {
          const int32_t clamped =
              std::min<int32_t>(127, std::max<int32_t>(-127, q_tmp[lane]));
          x_q[group_offset + e + lane] = static_cast<int8_t>(clamped);
        }
      }
    }
  }
}

void QuantizeActivationInt8(
    const void* HWY_RESTRICT x,
    QuantizedHighwayDType dtype,
    int K,
    int group_size,
    int8_t* HWY_RESTRICT x_q,
    float* HWY_RESTRICT x_scales,
    float* HWY_RESTRICT x_group_sums) {
  switch (dtype) {
    case QuantizedHighwayDType::Float32:
      QuantizeActivationInt8Typed(
          static_cast<const float*>(x),
          K,
          group_size,
          x_q,
          x_scales,
          x_group_sums);
      break;
    case QuantizedHighwayDType::Float16:
      QuantizeActivationInt8Typed(
          static_cast<const float16_t*>(x),
          K,
          group_size,
          x_q,
          x_scales,
          x_group_sums);
      break;
    case QuantizedHighwayDType::BFloat16:
      QuantizeActivationInt8Typed(
          static_cast<const bfloat16_t*>(x),
          K,
          group_size,
          x_q,
          x_scales,
          x_group_sums);
      break;
  }
}

template <typename T, int bits, int group_size>
void QmmTInt8RowTyped(
    float* HWY_RESTRICT result,
    const int8_t* HWY_RESTRICT x_q,
    const float* HWY_RESTRICT x_scales,
    const float* HWY_RESTRICT x_group_sums,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int n_start,
    int n_end,
    int K) {
  int out = 0;
  int n = n_start;
  if constexpr (!kHas512BitX86Vectors) {
    for (; n + 8 <= n_end; n += 8, out += 8) {
      qmm_t_int8_cols<float, T, bits, group_size, 8>(
          result + out, x_q, x_scales, x_group_sums, w, scales, biases, n, K);
    }
  }
  for (; n + 4 <= n_end; n += 4, out += 4) {
    qmm_t_int8_cols<float, T, bits, group_size, 4>(
        result + out, x_q, x_scales, x_group_sums, w, scales, biases, n, K);
  }
  for (; n < n_end; ++n, ++out) {
    qmm_t_int8_cols<float, T, bits, group_size, 1>(
        result + out, x_q, x_scales, x_group_sums, w, scales, biases, n, K);
  }
}

template <typename T, int bits>
void QmmTInt8RowForBits(
    float* HWY_RESTRICT result,
    const int8_t* HWY_RESTRICT x_q,
    const float* HWY_RESTRICT x_scales,
    const float* HWY_RESTRICT x_group_sums,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int group_size,
    int n_start,
    int n_end,
    int K) {
  switch (group_size) {
    case 32:
      QmmTInt8RowTyped<T, bits, 32>(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          n_start,
          n_end,
          K);
      break;
    case 64:
      QmmTInt8RowTyped<T, bits, 64>(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          n_start,
          n_end,
          K);
      break;
    case 128:
      QmmTInt8RowTyped<T, bits, 128>(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          n_start,
          n_end,
          K);
      break;
    default:
      throw std::invalid_argument(
          "Quantization group size must be 32, 64 or 128.");
  }
}

template <typename T>
void QmmTInt8RowForDType(
    float* HWY_RESTRICT result,
    const int8_t* HWY_RESTRICT x_q,
    const float* HWY_RESTRICT x_scales,
    const float* HWY_RESTRICT x_group_sums,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K) {
  switch (bits) {
    case 4:
      QmmTInt8RowForBits<T, 4>(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n_start,
          n_end,
          K);
      break;
    case 8:
      QmmTInt8RowForBits<T, 8>(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n_start,
          n_end,
          K);
      break;
    default:
      throw std::invalid_argument("Quantization bits must be 4 or 8.");
  }
}

void QmmTInt8Row(
    float* HWY_RESTRICT result,
    const int8_t* HWY_RESTRICT x_q,
    const float* HWY_RESTRICT x_scales,
    const float* HWY_RESTRICT x_group_sums,
    const uint32_t* HWY_RESTRICT w,
    const void* HWY_RESTRICT scales,
    const void* HWY_RESTRICT biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K) {
  switch (dtype) {
    case QuantizedHighwayDType::Float32:
      QmmTInt8RowForDType(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          static_cast<const float*>(scales),
          static_cast<const float*>(biases),
          bits,
          group_size,
          n_start,
          n_end,
          K);
      break;
    case QuantizedHighwayDType::Float16:
      QmmTInt8RowForDType(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          static_cast<const float16_t*>(scales),
          static_cast<const float16_t*>(biases),
          bits,
          group_size,
          n_start,
          n_end,
          K);
      break;
    case QuantizedHighwayDType::BFloat16:
      QmmTInt8RowForDType(
          result,
          x_q,
          x_scales,
          x_group_sums,
          w,
          static_cast<const bfloat16_t*>(scales),
          static_cast<const bfloat16_t*>(biases),
          bits,
          group_size,
          n_start,
          n_end,
          K);
      break;
  }
}

struct QmmTInt8Prequantized {
  const int8_t* x_q;
  const float* x_scales;
  const float* x_group_sums;
};

template <typename T, int bits, int group_size>
void QmmTInt8RowConverted(
    T* HWY_RESTRICT result,
    const int8_t* HWY_RESTRICT x_q,
    const float* HWY_RESTRICT x_scales,
    const float* HWY_RESTRICT x_group_sums,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int n_start,
    int n_end,
    int K) {
  int out = 0;
  int n = n_start;
  if constexpr (!kHas512BitX86Vectors) {
    for (; n + 8 <= n_end; n += 8, out += 8) {
      qmm_t_int8_cols<T, T, bits, group_size, 8>(
          result + out, x_q, x_scales, x_group_sums, w, scales, biases, n, K);
    }
  }
  for (; n + 4 <= n_end; n += 4, out += 4) {
    qmm_t_int8_cols<T, T, bits, group_size, 4>(
        result + out, x_q, x_scales, x_group_sums, w, scales, biases, n, K);
  }
  for (; n < n_end; ++n, ++out) {
    qmm_t_int8_cols<T, T, bits, group_size, 1>(
        result + out, x_q, x_scales, x_group_sums, w, scales, biases, n, K);
  }
}

template <typename T, int bits, int group_size>
void QmmTInt8RowFromActivation(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int n_start,
    int n_end,
    int K,
    const QmmTInt8Prequantized* preq) {
  constexpr int INT8_MAX_K = 16384;
  constexpr int STACK_GROUPS = 128;

  const int groups_per_col = K / group_size;
  alignas(64) int8_t x_q_stack[INT8_MAX_K];
  float x_scales_stack[STACK_GROUPS];
  float x_group_sums_stack[STACK_GROUPS];
  std::unique_ptr<float[]> x_scales_heap;
  std::unique_ptr<float[]> x_group_sums_heap;

  const int8_t* x_q = nullptr;
  const float* x_scales = nullptr;
  const float* x_group_sums = nullptr;
  if (preq) {
    x_q = preq->x_q;
    x_scales = preq->x_scales;
    x_group_sums = preq->x_group_sums;
  } else {
    float* x_scales_w = x_scales_stack;
    float* x_group_sums_w = x_group_sums_stack;
    if (groups_per_col > STACK_GROUPS) {
      x_scales_heap.reset(new float[groups_per_col]);
      x_group_sums_heap.reset(new float[groups_per_col]);
      x_scales_w = x_scales_heap.get();
      x_group_sums_w = x_group_sums_heap.get();
    }
    QuantizeActivationInt8Typed<T>(
        x, K, group_size, x_q_stack, x_scales_w, x_group_sums_w);
    x_q = x_q_stack;
    x_scales = x_scales_w;
    x_group_sums = x_group_sums_w;
  }

  QmmTInt8RowConverted<T, bits, group_size>(
      result,
      x_q,
      x_scales,
      x_group_sums,
      w,
      scales,
      biases,
      n_start,
      n_end,
      K);
}

template <typename T, int bits, int group_size>
bool QmmTInt8HighwayTyped(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int M,
    int N,
    int K) {
  constexpr int INT8_MAX_K = 16384;
  constexpr int STACK_GROUPS = 128;
  if (!env::enable_tf32() || K > INT8_MAX_K) {
    return false;
  }

  auto& pool = cpu::ThreadPool::instance();
  const int min_cols_per_thread = (M == 1) ? 128 : 64;
  const int n_threads =
      std::min(pool.max_threads(), std::max(1, N / min_cols_per_thread));
  constexpr int CHUNK_COLS = 64;
  const int n_chunks = (N + CHUNK_COLS - 1) / CHUNK_COLS;
  alignas(64) std::atomic<int> steal_counter{0};

  if (n_threads > 1 && M == 1) {
    const int groups_per_col = K / group_size;
    alignas(64) int8_t x_q[INT8_MAX_K];
    float x_scales_stack[STACK_GROUPS];
    float x_group_sums_stack[STACK_GROUPS];
    std::unique_ptr<float[]> x_scales_heap;
    std::unique_ptr<float[]> x_group_sums_heap;
    float* x_scales = x_scales_stack;
    float* x_group_sums = x_group_sums_stack;
    if (groups_per_col > STACK_GROUPS) {
      x_scales_heap.reset(new float[groups_per_col]);
      x_group_sums_heap.reset(new float[groups_per_col]);
      x_scales = x_scales_heap.get();
      x_group_sums = x_group_sums_heap.get();
    }

    QuantizeActivationInt8Typed<T>(
        x, K, group_size, x_q, x_scales, x_group_sums);
    QmmTInt8Prequantized preq{x_q, x_scales, x_group_sums};
    steal_counter.store(n_threads, std::memory_order_relaxed);
    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int my_chunk = tid;
      while (my_chunk < n_chunks) {
        const int n_start = std::min(my_chunk * CHUNK_COLS, N);
        const int n_end = std::min(n_start + CHUNK_COLS, N);
        if (n_start < n_end) {
          QmmTInt8RowFromActivation<T, bits, group_size>(
              result + n_start, x, w, scales, biases, n_start, n_end, K, &preq);
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
    return true;
  }

  if (n_threads > 1) {
    steal_counter.store(n_threads, std::memory_order_relaxed);
    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int my_chunk = tid;
      while (my_chunk < n_chunks) {
        const int n_start = std::min(my_chunk * CHUNK_COLS, N);
        const int n_end = std::min(n_start + CHUNK_COLS, N);
        if (n_start < n_end) {
          for (int m = 0; m < M; ++m) {
            QmmTInt8RowFromActivation<T, bits, group_size>(
                result + m * N + n_start,
                x + m * K,
                w,
                scales,
                biases,
                n_start,
                n_end,
                K,
                nullptr);
          }
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
  } else {
    for (int m = 0; m < M; ++m) {
      QmmTInt8RowFromActivation<T, bits, group_size>(
          result + m * N, x + m * K, w, scales, biases, 0, N, K, nullptr);
    }
  }
  return true;
}

template <typename T>
bool QmmTInt8HighwayForDType(
    void* HWY_RESTRICT result,
    const void* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const void* HWY_RESTRICT scales,
    const void* HWY_RESTRICT biases,
    int bits,
    int group_size,
    int M,
    int N,
    int K) {
  auto* result_t = static_cast<T*>(result);
  const auto* x_t = static_cast<const T*>(x);
  const auto* scales_t = static_cast<const T*>(scales);
  const auto* biases_t = static_cast<const T*>(biases);
  if (bits == 4) {
    if (group_size == 32) {
      return QmmTInt8HighwayTyped<T, 4, 32>(
          result_t, x_t, w, scales_t, biases_t, M, N, K);
    }
    if (group_size == 64) {
      return QmmTInt8HighwayTyped<T, 4, 64>(
          result_t, x_t, w, scales_t, biases_t, M, N, K);
    }
    if (group_size == 128) {
      return QmmTInt8HighwayTyped<T, 4, 128>(
          result_t, x_t, w, scales_t, biases_t, M, N, K);
    }
  } else if (bits == 8) {
    if (group_size == 32) {
      return QmmTInt8HighwayTyped<T, 8, 32>(
          result_t, x_t, w, scales_t, biases_t, M, N, K);
    }
    if (group_size == 64) {
      return QmmTInt8HighwayTyped<T, 8, 64>(
          result_t, x_t, w, scales_t, biases_t, M, N, K);
    }
    if (group_size == 128) {
      return QmmTInt8HighwayTyped<T, 8, 128>(
          result_t, x_t, w, scales_t, biases_t, M, N, K);
    }
  }
  throw std::invalid_argument(
      "Quantization bits must be 4 or 8 and group size must be 32, 64 or 128.");
}

bool QmmTInt8(
    void* HWY_RESTRICT result,
    const void* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const void* HWY_RESTRICT scales,
    const void* HWY_RESTRICT biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K) {
  switch (dtype) {
    case QuantizedHighwayDType::Float32:
      return QmmTInt8HighwayForDType<float>(
          result, x, w, scales, biases, bits, group_size, M, N, K);
    case QuantizedHighwayDType::Float16:
      return QmmTInt8HighwayForDType<float16_t>(
          result, x, w, scales, biases, bits, group_size, M, N, K);
    case QuantizedHighwayDType::BFloat16:
      return QmmTInt8HighwayForDType<bfloat16_t>(
          result, x, w, scales, biases, bits, group_size, M, N, K);
  }
  return false;
}

template <class DF>
hn::Vec<DF> load_fp4_weight_values(
    DF df,
    const uint32_t* w,
    int elem_off,
    const float* fp4_lut) {
  constexpr int pack_factor = 8;
  const size_t lanes = hn::Lanes(df);

  if constexpr (hn::MaxLanes(DF{}) == 16) {
    const hn::Half<DF> df_half;
    const uint32_t* words = w + elem_off * 2;
    const auto lo = load_fp4_weight_values(df_half, words, 0, fp4_lut);
    const auto hi = load_fp4_weight_values(df_half, words + 1, 0, fp4_lut);
    return hn::Combine(df, hi, lo);
  } else if constexpr (hn::MaxLanes(DF{}) == 8) {
    const hn::Rebind<uint32_t, DF> du32;
    const auto packed = hn::Set(du32, *w);
    const auto shifts = hn::Mul(
        hn::Iota(du32, static_cast<uint32_t>(elem_off * lanes)),
        hn::Set(du32, uint32_t{4}));
    const auto idx =
        hn::And(hn::Shr(packed, shifts), hn::Set(du32, uint32_t{0x0F}));
    const auto idx_lo = hn::And(idx, hn::Set(du32, uint32_t{0x07}));
    const auto low = hn::TableLookupLanes(
        hn::LoadU(df, fp4_lut), hn::IndicesFromVec(df, idx_lo));
    const auto high = hn::TableLookupLanes(
        hn::LoadU(df, fp4_lut + 8), hn::IndicesFromVec(df, idx_lo));
    const auto high_mask =
        hn::RebindMask(df, hn::Gt(idx, hn::Set(du32, uint32_t{7})));
    return hn::IfThenElse(high_mask, high, low);
  }

  alignas(64) float tmp[pack_factor];
  const uint32_t word = *w;
  const int base = elem_off * static_cast<int>(lanes);
  for (size_t lane = 0; lane < lanes; ++lane) {
    tmp[lane] = fp4_lut[(word >> ((base + static_cast<int>(lane)) * 4)) & 0xF];
  }
  return hn::LoadU(df, tmp);
}

template <class DF>
hn::Vec<DF> load_fp8_weight_values(
    DF df,
    const uint32_t* w,
    int elem_off,
    const float* fp8_lut) {
  const hn::Rebind<uint8_t, DF> du8;
  const hn::Rebind<uint16_t, DF> du16;
  const hn::Rebind<hwy::float16_t, DF> df16;
  const size_t lanes = hn::Lanes(df);
  const auto* bytes = reinterpret_cast<const uint8_t*>(w);
  const int base = elem_off * static_cast<int>(lanes);
  const auto encoded = hn::LoadU(du8, bytes + base);
  const auto encoded16 = hn::PromoteTo(du16, encoded);
  const auto payload =
      hn::ShiftLeft<7>(hn::And(encoded16, hn::Set(du16, 0x7F)));
  const auto sign = hn::ShiftLeft<8>(hn::And(encoded16, hn::Set(du16, 0x80)));
  return hn::Mul(
      hn::PromoteTo(df, hn::BitCast(df16, hn::Or(payload, sign))),
      hn::Set(df, 256.0f));
}

template <int group_size>
float dequantize_fp_scale(
    uint8_t encoded,
    float scale_factor,
    const float* fp8_lut) {
  if constexpr (group_size == 16) {
    return fp8_lut[encoded] * scale_factor;
  } else {
    union FOrI {
      bfloat16_t f;
      uint16_t i;
    } out;
    out.i = encoded == 0 ? 0x40 : (static_cast<uint16_t>(encoded) << 7);
    return static_cast<float>(out.f) * scale_factor;
  }
}

template <typename T, int NC, int group_size, int bits>
void fp_qmm_t_highway_cols(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* w_ptrs[NC],
    const uint8_t* scales_ptrs[NC],
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  static_assert(bits == 4 || bits == 8);
  static_assert(group_size == 16 || group_size == 32);

  constexpr int lane_cap = kHas512BitX86Vectors ? 16 : 8;
  const hn::CappedTag<float, lane_cap> df;
  using VF = hn::Vec<decltype(df)>;
  const int lanes = static_cast<int>(hn::Lanes(df));
  constexpr int pack_factor = 32 / bits;
  const int iters_per_word = (lanes >= pack_factor) ? 1 : pack_factor / lanes;
  const int words_per_iter = (lanes >= pack_factor) ? lanes / pack_factor : 0;

  VF acc[NC];
  HWY_UNROLL(4)
  for (int c = 0; c < NC; ++c) {
    acc[c] = hn::Zero(df);
  }

  for (int g = 0; g < K / group_size; ++g) {
    float scale_f[NC];
    HWY_UNROLL(4)
    for (int c = 0; c < NC; ++c) {
      scale_f[c] = dequantize_fp_scale<group_size>(
          *scales_ptrs[c]++, scale_factor, fp8_lut);
    }

    VF group_acc[NC];
    HWY_UNROLL(4)
    for (int c = 0; c < NC; ++c) {
      group_acc[c] = hn::Zero(df);
    }

    const size_t group_offset = static_cast<size_t>(g) * group_size;
    HWY_UNROLL(4)
    for (int elem = 0; elem < group_size; elem += lanes) {
      const VF x_vec = load_typed_as_f32(df, x, group_offset + elem);
      const int elem_off =
          (lanes >= pack_factor) ? 0 : (elem / lanes) % iters_per_word;

      HWY_UNROLL(4)
      for (int c = 0; c < NC; ++c) {
        VF w_vec;
        if constexpr (bits == 4) {
          w_vec = load_fp4_weight_values(df, w_ptrs[c], elem_off, fp4_lut);
        } else {
          w_vec = load_fp8_weight_values(df, w_ptrs[c], elem_off, fp8_lut);
        }
        group_acc[c] = hn::MulAdd(x_vec, w_vec, group_acc[c]);
        if (lanes >= pack_factor) {
          w_ptrs[c] += words_per_iter;
        } else if (elem_off == iters_per_word - 1) {
          w_ptrs[c] += 1;
        }
      }
    }

    HWY_UNROLL(4)
    for (int c = 0; c < NC; ++c) {
      acc[c] = hn::MulAdd(hn::Set(df, scale_f[c]), group_acc[c], acc[c]);
    }
  }

  HWY_UNROLL(4)
  for (int c = 0; c < NC; ++c) {
    result[c] = static_cast<T>(hn::ReduceSum(df, acc[c]));
  }
}

template <typename T, int group_size, int bits>
void fp_qmm_t_highway_cols_dispatch(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* w_ptrs[4],
    const uint8_t* scales_ptrs[4],
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  fp_qmm_t_highway_cols<T, 4, group_size, bits>(
      result, x, w_ptrs, scales_ptrs, K, scale_factor, fp4_lut, fp8_lut);
}

template <typename T, int group_size, int bits>
void fp_qmm_t_highway_col_dispatch(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* w_ptrs[1],
    const uint8_t* scales_ptrs[1],
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  fp_qmm_t_highway_cols<T, 1, group_size, bits>(
      result, x, w_ptrs, scales_ptrs, K, scale_factor, fp4_lut, fp8_lut);
}

template <typename T, int group_size, int bits>
void FpQmmTHighwayRowTyped(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const uint8_t* HWY_RESTRICT scales,
    int n_start,
    int n_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  const int pack_factor = 32 / bits;
  const int groups_per_col = K / group_size;
  const int packs_per_col = groups_per_col * (group_size / pack_factor);

  const uint32_t* w_base = w + n_start * packs_per_col;
  const uint8_t* scales_base = scales + n_start * groups_per_col;

  int out = 0;
  int n = n_start;
  for (; n + 4 <= n_end; n += 4, out += 4) {
    const uint32_t* wp[4];
    const uint8_t* sp[4];
    HWY_UNROLL(4)
    for (int c = 0; c < 4; ++c) {
      wp[c] = w_base + c * packs_per_col;
      sp[c] = scales_base + c * groups_per_col;
    }
    fp_qmm_t_highway_cols_dispatch<T, group_size, bits>(
        result + out, x, wp, sp, K, scale_factor, fp4_lut, fp8_lut);
    w_base += 4 * packs_per_col;
    scales_base += 4 * groups_per_col;
  }

  for (; n < n_end; ++n, ++out) {
    const uint32_t* wp[1] = {w_base};
    const uint8_t* sp[1] = {scales_base};
    fp_qmm_t_highway_col_dispatch<T, group_size, bits>(
        result + out, x, wp, sp, K, scale_factor, fp4_lut, fp8_lut);
    w_base += packs_per_col;
    scales_base += groups_per_col;
  }
}

template <typename T, int group_size, int bits>
void FpQmmTHighwayTyped(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const uint8_t* HWY_RESTRICT scales,
    int M,
    int N,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  auto& pool = cpu::ThreadPool::instance();

  const int min_cols_per_thread = (M == 1) ? 128 : 64;
  const int n_threads =
      std::min(pool.max_threads(), std::max(1, N / min_cols_per_thread));

  if (n_threads > 1) {
    constexpr int CHUNK_COLS = 64;
    const int n_chunks = (N + CHUNK_COLS - 1) / CHUNK_COLS;
    alignas(64) std::atomic<int> steal_counter{0};
    steal_counter.store(n_threads, std::memory_order_relaxed);

    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int my_chunk = tid;
      while (my_chunk < n_chunks) {
        const int n_start = std::min(my_chunk * CHUNK_COLS, N);
        const int n_end = std::min(n_start + CHUNK_COLS, N);
        if (n_start < n_end) {
          for (int m = 0; m < M; ++m) {
            FpQmmTHighwayRowTyped<T, group_size, bits>(
                result + m * N + n_start,
                x + m * K,
                w,
                scales,
                n_start,
                n_end,
                K,
                scale_factor,
                fp4_lut,
                fp8_lut);
          }
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
  } else {
    for (int m = 0; m < M; ++m) {
      FpQmmTHighwayRowTyped<T, group_size, bits>(
          result + m * N,
          x + m * K,
          w,
          scales,
          0,
          N,
          K,
          scale_factor,
          fp4_lut,
          fp8_lut);
    }
  }
}

template <typename T>
void FpQmmTHighwayRowForDType(
    void* HWY_RESTRICT result,
    const void* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const uint8_t* HWY_RESTRICT scales,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  auto* result_t = static_cast<T*>(result);
  const auto* x_t = static_cast<const T*>(x);
  if (bits == 8) {
    FpQmmTHighwayRowTyped<T, 32, 8>(
        result_t,
        x_t,
        w,
        scales,
        n_start,
        n_end,
        K,
        scale_factor,
        fp4_lut,
        fp8_lut);
  } else if (group_size == 16) {
    FpQmmTHighwayRowTyped<T, 16, 4>(
        result_t,
        x_t,
        w,
        scales,
        n_start,
        n_end,
        K,
        scale_factor,
        fp4_lut,
        fp8_lut);
  } else {
    FpQmmTHighwayRowTyped<T, 32, 4>(
        result_t,
        x_t,
        w,
        scales,
        n_start,
        n_end,
        K,
        scale_factor,
        fp4_lut,
        fp8_lut);
  }
}

template <typename T>
void FpQmmTHighwayForDType(
    void* HWY_RESTRICT result,
    const void* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const uint8_t* HWY_RESTRICT scales,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  auto* result_t = static_cast<T*>(result);
  const auto* x_t = static_cast<const T*>(x);
  if (bits == 8) {
    FpQmmTHighwayTyped<T, 32, 8>(
        result_t, x_t, w, scales, M, N, K, scale_factor, fp4_lut, fp8_lut);
  } else if (group_size == 16) {
    FpQmmTHighwayTyped<T, 16, 4>(
        result_t, x_t, w, scales, M, N, K, scale_factor, fp4_lut, fp8_lut);
  } else {
    FpQmmTHighwayTyped<T, 32, 4>(
        result_t, x_t, w, scales, M, N, K, scale_factor, fp4_lut, fp8_lut);
  }
}

void FpQmmTHighwayRow(
    void* HWY_RESTRICT result,
    const void* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const uint8_t* HWY_RESTRICT scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  switch (dtype) {
    case QuantizedHighwayDType::Float32:
      FpQmmTHighwayRowForDType<float>(
          result,
          x,
          w,
          scales,
          bits,
          group_size,
          n_start,
          n_end,
          K,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
    case QuantizedHighwayDType::Float16:
      FpQmmTHighwayRowForDType<float16_t>(
          result,
          x,
          w,
          scales,
          bits,
          group_size,
          n_start,
          n_end,
          K,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
    case QuantizedHighwayDType::BFloat16:
      FpQmmTHighwayRowForDType<bfloat16_t>(
          result,
          x,
          w,
          scales,
          bits,
          group_size,
          n_start,
          n_end,
          K,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
  }
}

void FpQmmTHighway(
    void* HWY_RESTRICT result,
    const void* HWY_RESTRICT x,
    const uint32_t* HWY_RESTRICT w,
    const uint8_t* HWY_RESTRICT scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  switch (dtype) {
    case QuantizedHighwayDType::Float32:
      FpQmmTHighwayForDType<float>(
          result,
          x,
          w,
          scales,
          bits,
          group_size,
          M,
          N,
          K,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
    case QuantizedHighwayDType::Float16:
      FpQmmTHighwayForDType<float16_t>(
          result,
          x,
          w,
          scales,
          bits,
          group_size,
          M,
          N,
          K,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
    case QuantizedHighwayDType::BFloat16:
      FpQmmTHighwayForDType<bfloat16_t>(
          result,
          x,
          w,
          scales,
          bits,
          group_size,
          M,
          N,
          K,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
  }
}

void DequantRow4Bit(
    const uint32_t* HWY_RESTRICT w_row,
    const float* HWY_RESTRICT scales_row,
    const float* HWY_RESTRICT biases_row,
    float* HWY_RESTRICT out,
    int group_size,
    int K) {
  dequant_row<4>(w_row, scales_row, biases_row, out, group_size, K);
}

void DequantRow8Bit(
    const uint32_t* HWY_RESTRICT w_row,
    const float* HWY_RESTRICT scales_row,
    const float* HWY_RESTRICT biases_row,
    float* HWY_RESTRICT out,
    int group_size,
    int K) {
  dequant_row<8>(w_row, scales_row, biases_row, out, group_size, K);
}

} // namespace
} // namespace HWY_NAMESPACE
} // namespace mlx::core
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace mlx::core {

#if defined(MLX_HIGHWAY_MANUAL_TARGET)

#ifndef MLX_HIGHWAY_TARGET_SUFFIX
#error "MLX_HIGHWAY_TARGET_SUFFIX must be defined for manual Highway targets"
#endif

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)
#define MLX_HIGHWAY_TARGET_FUNC(name) \
  MLX_HIGHWAY_CONCAT(name, MLX_HIGHWAY_TARGET_SUFFIX)

void MLX_HIGHWAY_TARGET_FUNC(dequant_row_highway_4bit)(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  HWY_STATIC_DISPATCH(DequantRow4Bit)
  (w_row, scales_row, biases_row, out, group_size, K);
}

void MLX_HIGHWAY_TARGET_FUNC(dequant_row_highway_8bit)(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  HWY_STATIC_DISPATCH(DequantRow8Bit)
  (w_row, scales_row, biases_row, out, group_size, K);
}

void MLX_HIGHWAY_TARGET_FUNC(quantize_activation_int8_highway)(
    const void* x,
    QuantizedHighwayDType dtype,
    int K,
    int group_size,
    int8_t* x_q,
    float* x_scales,
    float* x_group_sums) {
  HWY_STATIC_DISPATCH(QuantizeActivationInt8)
  (x, dtype, K, group_size, x_q, x_scales, x_group_sums);
}

void MLX_HIGHWAY_TARGET_FUNC(qmm_t_int8_highway_row)(
    float* result,
    const int8_t* x_q,
    const float* x_scales,
    const float* x_group_sums,
    const uint32_t* w,
    const void* scales,
    const void* biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K) {
  HWY_STATIC_DISPATCH(QmmTInt8Row)
  (result,
   x_q,
   x_scales,
   x_group_sums,
   w,
   scales,
   biases,
   dtype,
   bits,
   group_size,
   n_start,
   n_end,
   K);
}

bool MLX_HIGHWAY_TARGET_FUNC(qmm_t_int8_highway)(
    void* result,
    const void* x,
    const uint32_t* w,
    const void* scales,
    const void* biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K) {
  return HWY_STATIC_DISPATCH(QmmTInt8)(
      result, x, w, scales, biases, dtype, bits, group_size, M, N, K);
}

void MLX_HIGHWAY_TARGET_FUNC(fp_qmm_t_highway_row)(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_STATIC_DISPATCH(FpQmmTHighwayRow)
  (result,
   x,
   w,
   scales,
   dtype,
   bits,
   group_size,
   n_start,
   n_end,
   K,
   scale_factor,
   fp4_lut,
   fp8_lut);
}

void MLX_HIGHWAY_TARGET_FUNC(fp_qmm_t_highway)(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_STATIC_DISPATCH(FpQmmTHighway)
  (result,
   x,
   w,
   scales,
   dtype,
   bits,
   group_size,
   M,
   N,
   K,
   scale_factor,
   fp4_lut,
   fp8_lut);
}

#undef MLX_HIGHWAY_TARGET_FUNC
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

#else

HWY_EXPORT(DequantRow4Bit);
HWY_EXPORT(DequantRow8Bit);
HWY_EXPORT(QuantizeActivationInt8);
HWY_EXPORT(QmmTInt8Row);
HWY_EXPORT(QmmTInt8);
HWY_EXPORT(FpQmmTHighwayRow);
HWY_EXPORT(FpQmmTHighway);

void dequant_row_highway_4bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  HWY_DYNAMIC_DISPATCH(DequantRow4Bit)
  (w_row, scales_row, biases_row, out, group_size, K);
}

void dequant_row_highway_8bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  HWY_DYNAMIC_DISPATCH(DequantRow8Bit)
  (w_row, scales_row, biases_row, out, group_size, K);
}

void quantize_activation_int8_highway(
    const void* x,
    QuantizedHighwayDType dtype,
    int K,
    int group_size,
    int8_t* x_q,
    float* x_scales,
    float* x_group_sums) {
  HWY_DYNAMIC_DISPATCH(QuantizeActivationInt8)
  (x, dtype, K, group_size, x_q, x_scales, x_group_sums);
}

void qmm_t_int8_highway_row(
    float* result,
    const int8_t* x_q,
    const float* x_scales,
    const float* x_group_sums,
    const uint32_t* w,
    const void* scales,
    const void* biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K) {
  HWY_DYNAMIC_DISPATCH(QmmTInt8Row)
  (result,
   x_q,
   x_scales,
   x_group_sums,
   w,
   scales,
   biases,
   dtype,
   bits,
   group_size,
   n_start,
   n_end,
   K);
}

bool qmm_t_int8_highway(
    void* result,
    const void* x,
    const uint32_t* w,
    const void* scales,
    const void* biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K) {
  return HWY_DYNAMIC_DISPATCH(QmmTInt8)(
      result, x, w, scales, biases, dtype, bits, group_size, M, N, K);
}

void fp_qmm_t_highway_row(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_DYNAMIC_DISPATCH(FpQmmTHighwayRow)
  (result,
   x,
   w,
   scales,
   dtype,
   bits,
   group_size,
   n_start,
   n_end,
   K,
   scale_factor,
   fp4_lut,
   fp8_lut);
}

void fp_qmm_t_highway(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_DYNAMIC_DISPATCH(FpQmmTHighway)
  (result,
   x,
   w,
   scales,
   dtype,
   bits,
   group_size,
   M,
   N,
   K,
   scale_factor,
   fp4_lut,
   fp8_lut);
}

#endif

} // namespace mlx::core
#endif // HWY_ONCE
