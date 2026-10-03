// Copyright © 2026 Apple Inc.

// Normally this file is compiled directly and Highway emits its runtime
// dispatch targets. Native MSVC builds compile this file once per target with
// MLX_HIGHWAY_MANUAL_TARGET and MLX_HIGHWAY_TARGET_SUFFIX so the same kernels
// are emitted as one manually suffixed specialization.

#include <algorithm>
#include <array>
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
  constexpr int STACK_GROUPS = 128;

  const int groups_per_col = K / group_size;
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
    // Sized from K, never assumed to fit the stack: this writes K bytes, so a
    // fixed array here is a buffer overrun the moment a caller stops capping K.
    Int8ActivationScratch x_q_scratch(K);
    float* x_scales_w = x_scales_stack;
    float* x_group_sums_w = x_group_sums_stack;
    if (groups_per_col > STACK_GROUPS) {
      x_scales_heap.reset(new float[groups_per_col]);
      x_group_sums_heap.reset(new float[groups_per_col]);
      x_scales_w = x_scales_heap.get();
      x_group_sums_w = x_group_sums_heap.get();
    }
    QuantizeActivationInt8Typed<T>(
        x, K, group_size, x_q_scratch.data(), x_scales_w, x_group_sums_w);
    // The scratch must outlive the kernel call below, so finish the work here
    // rather than letting it fall out of scope at the end of this branch.
    QmmTInt8RowConverted<T, bits, group_size>(
        result,
        x_q_scratch.data(),
        x_scales_w,
        x_group_sums_w,
        w,
        scales,
        biases,
        n_start,
        n_end,
        K);
    return;
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
  constexpr int STACK_GROUPS = 128;
  if (!env::enable_tf32()) {
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
    // Quantized once per matmul and shared across threads; see the note on
    // Int8ActivationScratch for why K is not capped here.
    Int8ActivationScratch x_q_scratch(K);
    int8_t* x_q = x_q_scratch.data();
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

template <int group_size, int bits, bool half_avx2_fp4 = false>
std::array<float, 256> make_fp_scale_lut(
    float scale_factor,
    const float* fp8_lut) {
  std::array<float, 256> lut;
#if HWY_TARGET == HWY_AVX2
  if constexpr (half_avx2_fp4 && group_size == 16 && bits == 4) {
    // The single-row AVX2 FP4 decoder uses doubled values to avoid a multiply.
    scale_factor *= 0.5f;
  }
#endif
  for (int i = 0; i < 256; ++i) {
    lut[i] = dequantize_fp_scale<group_size>(i, scale_factor, fp8_lut);
  }
  return lut;
}

template <int group_size, int bits>
void fp_dequant_rows_typed(
    const uint32_t* HWY_RESTRICT w,
    const uint8_t* HWY_RESTRICT scales,
    float* HWY_RESTRICT out,
    int row_start,
    int row_end,
    int K,
    const float* scale_lut,
    const float* fp4_lut,
    const float* fp8_lut) {
  constexpr int pack_factor = 32 / bits;
  const int groups_per_row = K / group_size;
  const int packs_per_row = groups_per_row * (group_size / pack_factor);
  constexpr int lane_cap = kHas512BitX86Vectors ? 16 : 8;
  const hn::CappedTag<float, lane_cap> df;
  const int lanes = static_cast<int>(hn::Lanes(df));
  const int iters_per_word = lanes >= pack_factor ? 1 : pack_factor / lanes;
  const int words_per_iter = lanes >= pack_factor ? lanes / pack_factor : 0;

  for (int row = row_start; row < row_end; ++row) {
    const uint32_t* w_row = w + static_cast<size_t>(row) * packs_per_row;
    const uint8_t* scales_row = scales + static_cast<size_t>(row) * groups_per_row;
    float* out_row = out + static_cast<size_t>(row - row_start) * K;

    for (int g = 0; g < groups_per_row; ++g) {
      const auto scale = hn::Set(df, scale_lut[scales_row[g]]);
      const uint32_t* w_group = w_row + g * (group_size / pack_factor);
      const int out_offset = g * group_size;

      for (int elem = 0; elem < group_size; elem += lanes) {
        const int elem_off =
            lanes >= pack_factor ? 0 : (elem / lanes) % iters_per_word;
        const auto weights = bits == 4
            ? load_fp4_weight_values(df, w_group, elem_off, fp4_lut)
            : load_fp8_weight_values(df, w_group, elem_off, fp8_lut);
        hn::StoreU(hn::Mul(weights, scale), df, out_row + out_offset + elem);

        if (lanes >= pack_factor) {
          w_group += words_per_iter;
        } else if (elem_off == iters_per_word - 1) {
          ++w_group;
        }
      }
    }
  }
}

void FpDequantRows(
    const uint32_t* w,
    const uint8_t* scales,
    float* out,
    int bits,
    int group_size,
    int row_start,
    int row_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  float scale_lut[256];
  if (group_size == 16) {
    for (int i = 0; i < 256; ++i) {
      scale_lut[i] = dequantize_fp_scale<16>(i, scale_factor, fp8_lut);
    }
    fp_dequant_rows_typed<16, 4>(
        w, scales, out, row_start, row_end, K, scale_lut, fp4_lut, fp8_lut);
  } else {
    for (int i = 0; i < 256; ++i) {
      scale_lut[i] = dequantize_fp_scale<32>(i, scale_factor, fp8_lut);
    }
    if (bits == 8) {
      fp_dequant_rows_typed<32, 8>(
          w, scales, out, row_start, row_end, K, scale_lut, fp4_lut, fp8_lut);
    } else {
      fp_dequant_rows_typed<32, 4>(
          w, scales, out, row_start, row_end, K, scale_lut, fp4_lut, fp8_lut);
    }
  }
}

template <typename T, int NC, int group_size, int bits>
void fp_qmm_t_highway_cols(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* w_ptrs[NC],
    const uint8_t* scales_ptrs[NC],
    int K,
    float,
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
    VF scale_v[NC];
    HWY_UNROLL(4)
    for (int c = 0; c < NC; ++c) {
      scale_v[c] = hn::Set(df, fp8_lut[*scales_ptrs[c]++]);
    }

    VF group_acc[NC];
    HWY_UNROLL(4)
    for (int c = 0; c < NC; ++c) {
      group_acc[c] = hn::Zero(df);
    }

    const size_t group_offset = static_cast<size_t>(g) * group_size;
#if HWY_TARGET == HWY_AVX2
    if constexpr (bits == 4 && group_size == 16) {
      alignas(16) static constexpr int8_t e2m1_twice[16] = {
          0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
      const hn::CappedTag<int8_t, 16> di8;
      const hn::RebindToUnsigned<decltype(di8)> du8;
      const hn::Half<decltype(di8)> di8_half;
      const hn::Rebind<int32_t, decltype(df)> di32;
      const auto lut = hn::BitCast(du8, hn::Load(di8, e2m1_twice));
      const auto nibble_mask = hn::Set(du8, uint8_t{0x0F});
      const VF x_lo = load_typed_as_f32(df, x, group_offset);
      const VF x_hi = load_typed_as_f32(df, x, group_offset + lanes);

      HWY_UNROLL(4)
      for (int c = 0; c < NC; ++c) {
        const auto packed =
            hn::LoadN(du8, reinterpret_cast<const uint8_t*>(w_ptrs[c]), 8);
        const auto indices = hn::InterleaveLower(
            hn::And(packed, nibble_mask),
            hn::And(hn::ShiftRight<4>(packed), nibble_mask));
        const auto twice = hn::BitCast(di8, hn::TableLookupBytes(lut, indices));
        const auto w_lo = hn::ConvertTo(
            df, hn::PromoteTo(di32, hn::LowerHalf(di8_half, twice)));
        const auto w_hi = hn::ConvertTo(
            df, hn::PromoteTo(di32, hn::UpperHalf(di8_half, twice)));
        group_acc[c] = hn::MulAdd(x_lo, w_lo, group_acc[c]);
        group_acc[c] = hn::MulAdd(x_hi, w_hi, group_acc[c]);
        w_ptrs[c] += 2;
      }
    } else
#endif
    {
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
    }

    HWY_UNROLL(4)
    for (int c = 0; c < NC; ++c) {
      acc[c] = hn::MulAdd(scale_v[c], group_acc[c], acc[c]);
    }
  }

  HWY_UNROLL(4)
  for (int c = 0; c < NC; ++c) {
    result[c] = static_cast<T>(hn::ReduceSum(df, acc[c]));
  }
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
    fp_qmm_t_highway_cols<T, 4, group_size, bits>(
        result + out, x, wp, sp, K, scale_factor, fp4_lut, fp8_lut);
    w_base += 4 * packs_per_col;
    scales_base += 4 * groups_per_col;
  }

  for (; n < n_end; ++n, ++out) {
    const uint32_t* wp[1] = {w_base};
    const uint8_t* sp[1] = {scales_base};
    fp_qmm_t_highway_cols<T, 1, group_size, bits>(
        result + out, x, wp, sp, K, scale_factor, fp4_lut, fp8_lut);
    w_base += packs_per_col;
    scales_base += groups_per_col;
  }
}

// Defined below (FP-quant few-row QMM section).
template <typename T, int group_size, int bits>
void fp_qmm_t_fewrow_typed(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t ldx,
    size_t ldc,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut);

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
  auto scale_lut = M == 1
      ? make_fp_scale_lut<group_size, bits, true>(scale_factor, fp8_lut)
      : make_fp_scale_lut<group_size, bits>(scale_factor, fp8_lut);

  const int min_cols_per_thread = (M == 1) ? 128 : 64;
  const int n_threads =
      std::min(pool.max_threads(), std::max(1, N / min_cols_per_thread));

  // For M > 1, block rows so each packed weight column is decoded once per
  // 8-row register block instead of once per row; decode (unpack + LUT)
  // dominates the fp path. Row blocks of 64 keep the x panel L2-resident.
  constexpr int M_BLOCK = 64;
  auto run_cols = [&](int n_start, int n_end) {
    if (M == 1) {
      // The single-row kernel blocks 4 columns to amortize x loads.
      FpQmmTHighwayRowTyped<T, group_size, bits>(
          result + n_start,
          x,
          w,
          scales,
          n_start,
          n_end,
          K,
          1.0f,
          fp4_lut,
          scale_lut.data());
      return;
    }
    for (int m0 = 0; m0 < M; m0 += M_BLOCK) {
      const int m_rows = M - m0 < M_BLOCK ? M - m0 : M_BLOCK;
      fp_qmm_t_fewrow_typed<T, group_size, bits>(
          result + static_cast<size_t>(m0) * N,
          x + static_cast<size_t>(m0) * K,
          w,
          scales,
          m_rows,
          n_start,
          n_end,
          K,
          K,
          N,
          1.0f,
          fp4_lut,
          scale_lut.data());
    }
  };

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
          run_cols(n_start, n_end);
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
  } else {
    run_cols(0, N);
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
    auto scale_lut = make_fp_scale_lut<32, 8>(scale_factor, fp8_lut);
    FpQmmTHighwayRowTyped<T, 32, 8>(
        result_t,
        x_t,
        w,
        scales,
        n_start,
        n_end,
        K,
        1.0f,
        fp4_lut,
        scale_lut.data());
  } else if (group_size == 16) {
    auto scale_lut = make_fp_scale_lut<16, 4, true>(scale_factor, fp8_lut);
    FpQmmTHighwayRowTyped<T, 16, 4>(
        result_t,
        x_t,
        w,
        scales,
        n_start,
        n_end,
        K,
        1.0f,
        fp4_lut,
        scale_lut.data());
  } else {
    auto scale_lut = make_fp_scale_lut<32, 4>(scale_factor, fp8_lut);
    FpQmmTHighwayRowTyped<T, 32, 4>(
        result_t,
        x_t,
        w,
        scales,
        n_start,
        n_end,
        K,
        1.0f,
        fp4_lut,
        scale_lut.data());
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

// ---------------------- FP-quant few-row QMM ----------------------
// Computes result[m * ldc] += dot(x[m], dequant(w_col)) for MR rows of x
// against a single packed fp-quant weight column, decoding each weight
// chunk exactly once. The per-row kernel decodes the same column M times,
// which dominates small-M cost since fp decode (unpack + LUT) is far more
// expensive than the FMA. The group scale is folded into the decoded
// weight vector (identical math to dequantize-then-matmul).
template <typename T, int MR, int group_size, int bits>
void fp_qmm_t_highway_rows(
    T* HWY_RESTRICT result,
    const T* HWY_RESTRICT x,
    const uint32_t* w,
    const uint8_t* scales,
    int K,
    size_t ldx,
    size_t ldc,
    float,
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

  VF acc[MR];
  HWY_UNROLL(8)
  for (int m = 0; m < MR; ++m) {
    acc[m] = hn::Zero(df);
  }

  for (int g = 0; g < K / group_size; ++g) {
    const float scale_f = fp8_lut[*scales++];
    const VF scale_vec = hn::Set(df, scale_f);

    const size_t group_offset = static_cast<size_t>(g) * group_size;
    HWY_UNROLL(4)
    for (int elem = 0; elem < group_size; elem += lanes) {
      const int elem_off =
          (lanes >= pack_factor) ? 0 : (elem / lanes) % iters_per_word;
      VF w_vec;
      if constexpr (bits == 4) {
        w_vec = load_fp4_weight_values(df, w, elem_off, fp4_lut);
      } else {
        w_vec = load_fp8_weight_values(df, w, elem_off, fp8_lut);
      }
      w_vec = hn::Mul(w_vec, scale_vec);
      if (lanes >= pack_factor) {
        w += words_per_iter;
      } else if (elem_off == iters_per_word - 1) {
        w += 1;
      }

      HWY_UNROLL(8)
      for (int m = 0; m < MR; ++m) {
        const VF x_vec =
            load_typed_as_f32(df, x + m * ldx, group_offset + elem);
        acc[m] = hn::MulAdd(x_vec, w_vec, acc[m]);
      }
    }
  }

  HWY_UNROLL(8)
  for (int m = 0; m < MR; ++m) {
    result[m * ldc] = static_cast<T>(hn::ReduceSum(df, acc[m]));
  }
}

// Cover output columns [n_start, n_end) for m_rows rows of x, blocking rows
// into register-sized groups so each weight column is decoded at most
// ceil(m_rows / 8) times.
template <typename T, int group_size, int bits>
void fp_qmm_t_fewrow_typed(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t ldx,
    size_t ldc,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  constexpr int pack_factor = 32 / bits;
  const int packs_per_col = (K / group_size) * (group_size / pack_factor);
  const int groups_per_col = K / group_size;

  for (int n = n_start; n < n_end; ++n) {
    const uint32_t* w_col = w + static_cast<size_t>(n) * packs_per_col;
    const uint8_t* s_col = scales + static_cast<size_t>(n) * groups_per_col;
    int m = 0;
    while (m_rows - m >= 8) {
      fp_qmm_t_highway_rows<T, 8, group_size, bits>(
          result + m * ldc + n,
          x + m * ldx,
          w_col,
          s_col,
          K,
          ldx,
          ldc,
          scale_factor,
          fp4_lut,
          fp8_lut);
      m += 8;
    }
    if (m_rows - m >= 4) {
      fp_qmm_t_highway_rows<T, 4, group_size, bits>(
          result + m * ldc + n,
          x + m * ldx,
          w_col,
          s_col,
          K,
          ldx,
          ldc,
          scale_factor,
          fp4_lut,
          fp8_lut);
      m += 4;
    }
    if (m_rows - m >= 2) {
      fp_qmm_t_highway_rows<T, 2, group_size, bits>(
          result + m * ldc + n,
          x + m * ldx,
          w_col,
          s_col,
          K,
          ldx,
          ldc,
          scale_factor,
          fp4_lut,
          fp8_lut);
      m += 2;
    }
    if (m_rows - m >= 1) {
      fp_qmm_t_highway_rows<T, 1, group_size, bits>(
          result + m * ldc + n,
          x + m * ldx,
          w_col,
          s_col,
          K,
          ldx,
          ldc,
          scale_factor,
          fp4_lut,
          fp8_lut);
    }
  }
}

template <typename T>
void fp_qmm_t_fewrow_mode(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int bits,
    int group_size,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t ldx,
    size_t ldc,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  if (bits == 8) {
    auto scale_lut = make_fp_scale_lut<32, 8>(scale_factor, fp8_lut);
    fp_qmm_t_fewrow_typed<T, 32, 8>(
        result,
        x,
        w,
        scales,
        m_rows,
        n_start,
        n_end,
        K,
        ldx,
        ldc,
        1.0f,
        fp4_lut,
        scale_lut.data());
  } else if (group_size == 32) {
    auto scale_lut = make_fp_scale_lut<32, 4>(scale_factor, fp8_lut);
    fp_qmm_t_fewrow_typed<T, 32, 4>(
        result,
        x,
        w,
        scales,
        m_rows,
        n_start,
        n_end,
        K,
        ldx,
        ldc,
        1.0f,
        fp4_lut,
        scale_lut.data());
  } else {
    auto scale_lut = make_fp_scale_lut<16, 4>(scale_factor, fp8_lut);
    fp_qmm_t_fewrow_typed<T, 16, 4>(
        result,
        x,
        w,
        scales,
        m_rows,
        n_start,
        n_end,
        K,
        ldx,
        ldc,
        1.0f,
        fp4_lut,
        scale_lut.data());
  }
}

void FpQmmTFewRow(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t ldx,
    size_t ldc,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  switch (dtype) {
    case QuantizedHighwayDType::Float32:
      fp_qmm_t_fewrow_mode(
          static_cast<float*>(result),
          static_cast<const float*>(x),
          w,
          scales,
          bits,
          group_size,
          m_rows,
          n_start,
          n_end,
          K,
          ldx,
          ldc,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
    case QuantizedHighwayDType::Float16:
      fp_qmm_t_fewrow_mode(
          static_cast<float16_t*>(result),
          static_cast<const float16_t*>(x),
          w,
          scales,
          bits,
          group_size,
          m_rows,
          n_start,
          n_end,
          K,
          ldx,
          ldc,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
    case QuantizedHighwayDType::BFloat16:
      fp_qmm_t_fewrow_mode(
          static_cast<bfloat16_t*>(result),
          static_cast<const bfloat16_t*>(x),
          w,
          scales,
          bits,
          group_size,
          m_rows,
          n_start,
          n_end,
          K,
          ldx,
          ldc,
          scale_factor,
          fp4_lut,
          fp8_lut);
      break;
  }
}

// ---------------------- Low-precision few-row GEMV ----------------------
// Computes out[m][n] = alpha * dot(a[m], b[n]) (+ beta * out[m][n]) for the
// output columns [n_start, n_end) of a GEMM against transposed B. Weights
// (B) are the dominant memory stream for few-row shapes, so each register
// block of MR rows streams B exactly once, converting in-register. This
// replaces the convert-everything-to-f32-then-BLAS path that inflates
// weight traffic (2x for halves) and previously ran single-threaded for
// M < 16.
template <typename AT, typename BT, typename OT, int MR>
void lowp_gemv_rows_typed(
    OT* HWY_RESTRICT out,
    const AT* HWY_RESTRICT a,
    const BT* HWY_RESTRICT b,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  const hn::ScalableTag<float> df;
  const size_t W = hn::Lanes(df);

  for (int n = n_start; n < n_end; n++) {
    const BT* HWY_RESTRICT b_row = b + static_cast<size_t>(n) * ldb;
    hn::Vec<decltype(df)> acc[MR];
    for (int m = 0; m < MR; m++) {
      acc[m] = hn::Zero(df);
    }
    size_t k = 0;
    for (; k + W <= static_cast<size_t>(K); k += W) {
      const auto bv = hu::load_typed_as_f32(df, b_row, k);
      for (int m = 0; m < MR; m++) {
        const auto av = hu::load_typed_as_f32(df, a + m * lda, k);
        acc[m] = hn::MulAdd(av, bv, acc[m]);
      }
    }
    float sums[MR];
    for (int m = 0; m < MR; m++) {
      sums[m] = hn::ReduceSum(df, acc[m]);
    }
    for (; k < static_cast<size_t>(K); k++) {
      const float bk = static_cast<float>(b_row[k]);
      for (int m = 0; m < MR; m++) {
        sums[m] += static_cast<float>(a[m * lda + k]) * bk;
      }
    }
    for (int m = 0; m < MR; m++) {
      float r = alpha * sums[m];
      if (beta != 0.0f) {
        r += beta * static_cast<float>(out[m * ldc + n]);
      }
      out[m * ldc + n] = static_cast<OT>(r);
    }
  }
}

template <typename AT, typename BT, typename OT, int NC>
void lowp_gemv_1row_cols_typed(
    OT* HWY_RESTRICT out,
    const AT* HWY_RESTRICT a,
    const BT* HWY_RESTRICT b,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  const hn::ScalableTag<float> df;
  const size_t W = hn::Lanes(df);
  for (int n = n_start; n < n_end; n += NC) {
    const int cols = std::min(NC, n_end - n);
    hn::Vec<decltype(df)> acc[NC];
    HWY_UNROLL(4)
    for (int c = 0; c < NC; c++) {
      acc[c] = hn::Zero(df);
    }
    size_t k = 0;
    for (; k + W <= static_cast<size_t>(K); k += W) {
      const auto av = hu::load_typed_as_f32(df, a, k);
      HWY_UNROLL(4)
      for (int c = 0; c < NC; c++) {
        if (c < cols) {
          const BT* HWY_RESTRICT b_row = b + static_cast<size_t>(n + c) * ldb;
          const auto bv = hu::load_typed_as_f32(df, b_row, k);
          acc[c] = hn::MulAdd(av, bv, acc[c]);
        }
      }
    }
    float sums[NC];
    HWY_UNROLL(4)
    for (int c = 0; c < NC; c++) {
      sums[c] = hn::ReduceSum(df, acc[c]);
    }
    for (; k < static_cast<size_t>(K); k++) {
      const float av = static_cast<float>(a[k]);
      for (int c = 0; c < cols; c++) {
        const BT* HWY_RESTRICT b_row = b + static_cast<size_t>(n + c) * ldb;
        sums[c] += av * static_cast<float>(b_row[k]);
      }
    }
    for (int c = 0; c < cols; c++) {
      float r = alpha * sums[c];
      if (beta != 0.0f) {
        r += beta * static_cast<float>(out[n + c]);
      }
      out[n + c] = static_cast<OT>(r);
    }
  }
}

template <typename AT, typename BT, typename OT>
void lowp_gemv_fewrow_typed(
    OT* out,
    const AT* a,
    const BT* b,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  if (m_rows == 1) {
    lowp_gemv_1row_cols_typed<AT, BT, OT, 2>(
        out, a, b, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
    return;
  }
  // Column-chunk the B panel so it stays cache-resident while every
  // register block of rows passes over it; otherwise each 8-row block
  // re-streams B from memory and throughput stops scaling with M.
  constexpr int N_CHUNK = 64;
  for (int nc = n_start; nc < n_end; nc += N_CHUNK) {
    const int nc_end = nc + N_CHUNK < n_end ? nc + N_CHUNK : n_end;
    int m = 0;
    while (m_rows - m >= 8) {
      lowp_gemv_rows_typed<AT, BT, OT, 8>(
          out + m * ldc,
          a + m * lda,
          b,
          nc,
          nc_end,
          K,
          lda,
          ldb,
          ldc,
          alpha,
          beta);
      m += 8;
    }
    if (m_rows - m >= 4) {
      lowp_gemv_rows_typed<AT, BT, OT, 4>(
          out + m * ldc,
          a + m * lda,
          b,
          nc,
          nc_end,
          K,
          lda,
          ldb,
          ldc,
          alpha,
          beta);
      m += 4;
    }
    if (m_rows - m >= 2) {
      lowp_gemv_rows_typed<AT, BT, OT, 2>(
          out + m * ldc,
          a + m * lda,
          b,
          nc,
          nc_end,
          K,
          lda,
          ldb,
          ldc,
          alpha,
          beta);
      m += 2;
    }
    if (m_rows - m >= 1) {
      lowp_gemv_rows_typed<AT, BT, OT, 1>(
          out + m * ldc,
          a + m * lda,
          b,
          nc,
          nc_end,
          K,
          lda,
          ldb,
          ldc,
          alpha,
          beta);
    }
  }
}

void LowpGemvFewRow(
    void* out,
    const void* a,
    const void* b,
    QuantizedHighwayDType dtype,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  switch (dtype) {
    case QuantizedHighwayDType::Float32:
      lowp_gemv_fewrow_typed<float, float, float>(
          static_cast<float*>(out),
          static_cast<const float*>(a),
          static_cast<const float*>(b),
          m_rows,
          n_start,
          n_end,
          K,
          lda,
          ldb,
          ldc,
          alpha,
          beta);
      break;
    case QuantizedHighwayDType::Float16:
      lowp_gemv_fewrow_typed<float16_t, float16_t, float16_t>(
          static_cast<float16_t*>(out),
          static_cast<const float16_t*>(a),
          static_cast<const float16_t*>(b),
          m_rows,
          n_start,
          n_end,
          K,
          lda,
          ldb,
          ldc,
          alpha,
          beta);
      break;
    case QuantizedHighwayDType::BFloat16:
      lowp_gemv_fewrow_typed<bfloat16_t, bfloat16_t, bfloat16_t>(
          static_cast<bfloat16_t*>(out),
          static_cast<const bfloat16_t*>(a),
          static_cast<const bfloat16_t*>(b),
          m_rows,
          n_start,
          n_end,
          K,
          lda,
          ldb,
          ldc,
          alpha,
          beta);
      break;
  }
}

void MixedLowpGemvFewRow(
    void* out,
    const void* a,
    const void* b,
    QuantizedHighwayDType b_dtype,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  switch (b_dtype) {
    case QuantizedHighwayDType::Float16:
      lowp_gemv_fewrow_typed<float, float16_t, float>(
          static_cast<float*>(out),
          static_cast<const float*>(a),
          static_cast<const float16_t*>(b),
          m_rows, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
      break;
    case QuantizedHighwayDType::BFloat16:
      lowp_gemv_fewrow_typed<float, bfloat16_t, float>(
          static_cast<float*>(out),
          static_cast<const float*>(a),
          static_cast<const bfloat16_t*>(b),
          m_rows, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
      break;
    case QuantizedHighwayDType::Float32:
      lowp_gemv_fewrow_typed<float, float, float>(
          static_cast<float*>(out),
          static_cast<const float*>(a),
          static_cast<const float*>(b),
          m_rows, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
      break;
  }
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

void MLX_HIGHWAY_TARGET_FUNC(fp_dequant_rows_highway)(
    const uint32_t* w,
    const uint8_t* scales,
    float* out,
    int bits,
    int group_size,
    int row_start,
    int row_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_STATIC_DISPATCH(FpDequantRows)
  (w,
   scales,
   out,
   bits,
   group_size,
   row_start,
   row_end,
   K,
   scale_factor,
   fp4_lut,
   fp8_lut);
}

void MLX_HIGHWAY_TARGET_FUNC(lowp_gemv_fewrow_highway)(
    void* out,
    const void* a,
    const void* b,
    QuantizedHighwayDType dtype,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  HWY_STATIC_DISPATCH(LowpGemvFewRow)
  (out, a, b, dtype, m_rows, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
}

void MLX_HIGHWAY_TARGET_FUNC(mixed_lowp_gemv_fewrow_highway)(
    void* out,
    const void* a,
    const void* b,
    QuantizedHighwayDType b_dtype,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  HWY_STATIC_DISPATCH(MixedLowpGemvFewRow)
  (out, a, b, b_dtype, m_rows, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
}

void MLX_HIGHWAY_TARGET_FUNC(fp_qmm_t_fewrow_highway)(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t ldx,
    size_t ldc,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_STATIC_DISPATCH(FpQmmTFewRow)
  (result,
   x,
   w,
   scales,
   dtype,
   bits,
   group_size,
   m_rows,
   n_start,
   n_end,
   K,
   ldx,
   ldc,
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
HWY_EXPORT(FpDequantRows);
HWY_EXPORT(LowpGemvFewRow);
HWY_EXPORT(MixedLowpGemvFewRow);
HWY_EXPORT(FpQmmTFewRow);

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

void fp_dequant_rows_highway(
    const uint32_t* w,
    const uint8_t* scales,
    float* out,
    int bits,
    int group_size,
    int row_start,
    int row_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_DYNAMIC_DISPATCH(FpDequantRows)
  (w,
   scales,
   out,
   bits,
   group_size,
   row_start,
   row_end,
   K,
   scale_factor,
   fp4_lut,
   fp8_lut);
}

void lowp_gemv_fewrow_highway(
    void* out,
    const void* a,
    const void* b,
    QuantizedHighwayDType dtype,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  HWY_DYNAMIC_DISPATCH(LowpGemvFewRow)
  (out, a, b, dtype, m_rows, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
}

void mixed_lowp_gemv_fewrow_highway(
    void* out,
    const void* a,
    const void* b,
    QuantizedHighwayDType b_dtype,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  HWY_DYNAMIC_DISPATCH(MixedLowpGemvFewRow)
  (out, a, b, b_dtype, m_rows, n_start, n_end, K, lda, ldb, ldc, alpha, beta);
}

void fp_qmm_t_fewrow_highway(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int m_rows,
    int n_start,
    int n_end,
    int K,
    size_t ldx,
    size_t ldc,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  HWY_DYNAMIC_DISPATCH(FpQmmTFewRow)
  (result,
   x,
   w,
   scales,
   dtype,
   bits,
   group_size,
   m_rows,
   n_start,
   n_end,
   K,
   ldx,
   ldc,
   scale_factor,
   fp4_lut,
   fp8_lut);
}

#endif

} // namespace mlx::core
#endif // HWY_ONCE
