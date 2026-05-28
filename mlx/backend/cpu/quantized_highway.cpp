// Copyright © 2026 Apple Inc.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/quantized_highway.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep

#include "mlx/backend/cpu/quantized_highway.h"

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T, class DF>
hn::Vec<DF> load_typed_as_f32(DF df, const T* HWY_RESTRICT ptr, size_t idx) {
  if constexpr (std::is_same_v<T, float>) {
    return hn::LoadU(df, ptr + idx);
  } else if constexpr (std::is_same_v<T, float16_t>) {
    const hn::Rebind<hwy::float16_t, DF> df16;
    return hn::PromoteTo(
        df,
        hn::LoadU(df16, reinterpret_cast<const hwy::float16_t*>(ptr) + idx));
  } else {
#if HWY_TARGET == HWY_SCALAR
    const hn::Rebind<hwy::bfloat16_t, DF> dbf16;
#else
    const hn::Repartition<hwy::bfloat16_t, DF> dbf16;
#endif
    const hn::Half<decltype(dbf16)> dbf16_half;
    return hn::PromoteTo(
        df,
        hn::LoadU(
            dbf16_half, reinterpret_cast<const hwy::bfloat16_t*>(ptr) + idx));
  }
}

template <class DU8>
hn::Vec<DU8> unpack_4bit_lanes(DU8 du8, const uint32_t* words) {
  const auto packed = hn::LoadN(
      du8, reinterpret_cast<const uint8_t*>(words), hn::Lanes(du8) / 2);
  const auto mask = hn::Set(du8, uint8_t{0x0F});
  const auto lo = hn::And(packed, mask);
  const auto hi = hn::And(hn::ShiftRight<4>(packed), mask);
  return hn::ConcatLowerLower(
      du8, hn::InterleaveUpper(du8, lo, hi), hn::InterleaveLower(lo, hi));
}

template <typename T, int bits, int NC>
void qmm_t_int8_cols(
    float* HWY_RESTRICT result,
    const int8_t* HWY_RESTRICT x_q,
    const float* HWY_RESTRICT x_scales,
    const float* HWY_RESTRICT x_group_sums,
    const uint32_t* HWY_RESTRICT w,
    const T* HWY_RESTRICT scales,
    const T* HWY_RESTRICT biases,
    int group_size,
    int n,
    int K) {
  static_assert(bits == 4 || bits == 8);

  constexpr int pack_factor = 32 / bits;
  const int groups_per_col = K / group_size;
  const int packs_in_group = group_size / pack_factor;
  const int packs_per_col = groups_per_col * packs_in_group;

  const hn::CappedTag<int8_t, 32> di8;
  const hn::RebindToUnsigned<decltype(di8)> du8;
  const hn::Repartition<int16_t, decltype(di8)> di16;
  const hn::Repartition<int32_t, decltype(di8)> di32;
  const size_t lanes = hn::Lanes(di8);
  const auto ones16 = hn::Set(di16, int16_t{1});

  float accum[NC] = {};

  for (int g = 0; g < groups_per_col; ++g) {
    hn::Vec<decltype(di32)> dot_acc[NC];
    for (int c = 0; c < NC; ++c) {
      dot_acc[c] = hn::Zero(di32);
    }

    const int8_t* x_group = x_q + g * group_size;
    for (int elem = 0; elem < group_size; elem += static_cast<int>(lanes)) {
      const auto x_vec = hn::LoadU(di8, x_group + elem);
      for (int c = 0; c < NC; ++c) {
        const uint32_t* w_group =
            w + (n + c) * packs_per_col + g * packs_in_group;
        hn::Vec<decltype(du8)> w_vec;
        if constexpr (bits == 4) {
          w_vec = unpack_4bit_lanes(du8, w_group + elem / pack_factor);
          const auto prod16 = hn::SatWidenMulPairwiseAdd(di16, w_vec, x_vec);
          dot_acc[c] = hn::Add(
              dot_acc[c], hn::WidenMulPairwiseAdd(di32, prod16, ones16));
        } else {
          const auto* w_bytes =
              reinterpret_cast<const uint8_t*>(w_group) + elem;
          w_vec = hn::LoadU(du8, w_bytes);
          const auto low7 = hn::And(w_vec, hn::Set(du8, uint8_t{0x7F}));
          const auto high_bit = hn::ShiftRight<7>(w_vec);
          const auto low16 = hn::SatWidenMulPairwiseAdd(di16, low7, x_vec);
          const auto high16 = hn::SatWidenMulPairwiseAdd(di16, high_bit, x_vec);
          dot_acc[c] = hn::Add(
              dot_acc[c],
              hn::Add(
                  hn::WidenMulPairwiseAdd(di32, low16, ones16),
                  hn::ShiftLeft<7>(
                      hn::WidenMulPairwiseAdd(di32, high16, ones16))));
        }
      }
    }

    const float xs = x_scales[g];
    const float xgs = x_group_sums[g];
    for (int c = 0; c < NC; ++c) {
      const size_t param_idx = static_cast<size_t>(n + c) * groups_per_col + g;
      const float scale_f = static_cast<float>(scales[param_idx]);
      const float bias_f = static_cast<float>(biases[param_idx]);
      const float dot = static_cast<float>(hn::ReduceSum(di32, dot_acc[c]));
      accum[c] += scale_f * xs * dot + bias_f * xgs;
    }
  }

  for (int c = 0; c < NC; ++c) {
    result[c] = accum[c];
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
  constexpr int lane_cap = bits == 4 ? 8 : 4;

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

    if (lanes == 8) {
      for (int e = 0; e < group_size; e += 32) {
        const auto q0 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e), inv));
        const auto q1 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 8), inv));
        const auto q2 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 16), inv));
        const auto q3 = hn::NearestInt(
            hn::Mul(load_typed_as_f32(df, x, group_offset + e + 24), inv));
        hn::StoreU(
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

template <typename T>
void QmmTInt8RowTyped(
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
  int out = 0;
  int n = n_start;
  for (; n + 8 <= n_end; n += 8, out += 8) {
    if (bits == 4) {
      qmm_t_int8_cols<T, 4, 8>(
          result + out,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n,
          K);
    } else {
      qmm_t_int8_cols<T, 8, 8>(
          result + out,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n,
          K);
    }
  }
  for (; n + 4 <= n_end; n += 4, out += 4) {
    if (bits == 4) {
      qmm_t_int8_cols<T, 4, 4>(
          result + out,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n,
          K);
    } else {
      qmm_t_int8_cols<T, 8, 4>(
          result + out,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n,
          K);
    }
  }
  for (; n < n_end; ++n, ++out) {
    if (bits == 4) {
      qmm_t_int8_cols<T, 4, 1>(
          result + out,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n,
          K);
    } else {
      qmm_t_int8_cols<T, 8, 1>(
          result + out,
          x_q,
          x_scales,
          x_group_sums,
          w,
          scales,
          biases,
          group_size,
          n,
          K);
    }
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
      QmmTInt8RowTyped(
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
      QmmTInt8RowTyped(
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
      QmmTInt8RowTyped(
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

template <class DF>
hn::Vec<DF> load_fp4_weight_values(
    DF df,
    const uint32_t* w,
    int elem_off,
    const float* fp4_lut) {
  constexpr int pack_factor = 8;
  const size_t lanes = hn::Lanes(df);

  const hn::Rebind<uint32_t, DF> du32;
  if (lanes == 8) {
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
  const hn::Rebind<int32_t, DF> di32;
  const hn::Rebind<uint8_t, DF> du8;
  const size_t lanes = hn::Lanes(df);
  const auto* bytes = reinterpret_cast<const uint8_t*>(w);
  const int base = elem_off * static_cast<int>(lanes);
  const auto indices = hn::PromoteTo(di32, hn::LoadU(du8, bytes + base));
  return hn::GatherIndex(df, fp8_lut, indices);
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

  const hn::CappedTag<float, 8> df;
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

HWY_EXPORT(DequantRow4Bit);
HWY_EXPORT(DequantRow8Bit);
HWY_EXPORT(QuantizeActivationInt8);
HWY_EXPORT(QmmTInt8Row);
HWY_EXPORT(FpQmmTHighwayRow);

void dequant_row_highway_4bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  HWY_DYNAMIC_DISPATCH(DequantRow4Bit)(
      w_row, scales_row, biases_row, out, group_size, K);
}

void dequant_row_highway_8bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  HWY_DYNAMIC_DISPATCH(DequantRow8Bit)(
      w_row, scales_row, biases_row, out, group_size, K);
}

void quantize_activation_int8_highway(
    const void* x,
    QuantizedHighwayDType dtype,
    int K,
    int group_size,
    int8_t* x_q,
    float* x_scales,
    float* x_group_sums) {
  HWY_DYNAMIC_DISPATCH(QuantizeActivationInt8)(
      x, dtype, K, group_size, x_q, x_scales, x_group_sums);
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
  HWY_DYNAMIC_DISPATCH(QmmTInt8Row)(
      result,
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
  HWY_DYNAMIC_DISPATCH(FpQmmTHighwayRow)(
      result,
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

} // namespace mlx::core
#endif // HWY_ONCE
