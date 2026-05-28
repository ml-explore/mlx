// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/rope_highway.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/rope_highway.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core::fast {
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

template <typename T, class DF>
void store_f32_as_typed(DF df, hn::Vec<DF> v, T* HWY_RESTRICT ptr, size_t idx) {
  if constexpr (std::is_same_v<T, float>) {
    hn::StoreU(v, df, ptr + idx);
  } else if constexpr (std::is_same_v<T, float16_t>) {
    const hn::Rebind<hwy::float16_t, DF> df16;
    hn::StoreU(
        hn::DemoteTo(df16, v),
        df16,
        reinterpret_cast<hwy::float16_t*>(ptr) + idx);
  } else {
#if HWY_TARGET == HWY_SCALAR
    const hn::Rebind<hwy::bfloat16_t, DF> dbf16;
#else
    const hn::Repartition<hwy::bfloat16_t, DF> dbf16;
#endif
    const hn::Half<decltype(dbf16)> dbf16_half;
    hn::StoreU(
        hn::DemoteTo(dbf16_half, v),
        dbf16_half,
        reinterpret_cast<hwy::bfloat16_t*>(ptr) + idx);
  }
}

template <typename T, class DF>
void load_interleaved_typed_as_f32(
    DF df,
    const T* HWY_RESTRICT ptr,
    size_t idx,
    hn::Vec<DF>& x0,
    hn::Vec<DF>& x1) {
  if constexpr (std::is_same_v<T, float>) {
    hn::LoadInterleaved2(df, ptr + idx, x0, x1);
  } else if constexpr (std::is_same_v<T, float16_t>) {
    const hn::Rebind<hwy::float16_t, DF> df16;
    hn::Vec<decltype(df16)> x0h;
    hn::Vec<decltype(df16)> x1h;
    hn::LoadInterleaved2(
        df16, reinterpret_cast<const hwy::float16_t*>(ptr) + idx, x0h, x1h);
    x0 = hn::PromoteTo(df, x0h);
    x1 = hn::PromoteTo(df, x1h);
  } else {
#if HWY_TARGET == HWY_SCALAR
    const hn::Rebind<hwy::bfloat16_t, DF> dbf16;
#else
    const hn::Repartition<hwy::bfloat16_t, DF> dbf16;
#endif
    const hn::Half<decltype(dbf16)> dbf16_half;
    hn::Vec<decltype(dbf16_half)> x0h;
    hn::Vec<decltype(dbf16_half)> x1h;
    hn::LoadInterleaved2(
        dbf16_half,
        reinterpret_cast<const hwy::bfloat16_t*>(ptr) + idx,
        x0h,
        x1h);
    x0 = hn::PromoteTo(df, x0h);
    x1 = hn::PromoteTo(df, x1h);
  }
}

template <typename T, class DF>
void store_interleaved_f32_as_typed(
    DF df,
    hn::Vec<DF> out0,
    hn::Vec<DF> out1,
    T* HWY_RESTRICT ptr,
    size_t idx) {
  if constexpr (std::is_same_v<T, float>) {
    hn::StoreInterleaved2(out0, out1, df, ptr + idx);
  } else if constexpr (std::is_same_v<T, float16_t>) {
    const hn::Rebind<hwy::float16_t, DF> df16;
    hn::StoreInterleaved2(
        hn::DemoteTo(df16, out0),
        hn::DemoteTo(df16, out1),
        df16,
        reinterpret_cast<hwy::float16_t*>(ptr) + idx);
  } else {
#if HWY_TARGET == HWY_SCALAR
    const hn::Rebind<hwy::bfloat16_t, DF> dbf16;
#else
    const hn::Repartition<hwy::bfloat16_t, DF> dbf16;
#endif
    const hn::Half<decltype(dbf16)> dbf16_half;
    hn::StoreInterleaved2(
        hn::DemoteTo(dbf16_half, out0),
        hn::DemoteTo(dbf16_half, out1),
        dbf16_half,
        reinterpret_cast<hwy::bfloat16_t*>(ptr) + idx);
  }
}

template <typename T, bool forward>
int rope_traditional(
    const T* HWY_RESTRICT x_in,
    T* HWY_RESTRICT x_out,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));

  int j = 0;
  HWY_UNROLL(4)
  for (; j + lanes <= half_dims; j += lanes) {
    V x0;
    V x1;
    load_interleaved_typed_as_f32(d, x_in, 2 * j, x0, x1);

    const V c = hn::LoadU(d, cos_t + j);
    const V s = hn::LoadU(d, sin_t + j);

    V out0;
    V out1;
    if constexpr (forward) {
      out0 = hn::Sub(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Add(hn::Mul(x0, s), hn::Mul(x1, c));
    } else {
      out0 = hn::Add(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Sub(hn::Mul(x1, c), hn::Mul(x0, s));
    }

    store_interleaved_f32_as_typed(d, out0, out1, x_out, 2 * j);
  }
  return j;
}

template <typename T, bool forward>
int rope_non_traditional(
    const T* HWY_RESTRICT x_in,
    T* HWY_RESTRICT x_out,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));

  int j = 0;
  HWY_UNROLL(4)
  for (; j + lanes <= half_dims; j += lanes) {
    const V x0 = load_typed_as_f32(d, x_in, j);
    const V x1 = load_typed_as_f32(d, x_in, j + half_dims);
    const V c = hn::LoadU(d, cos_t + j);
    const V s = hn::LoadU(d, sin_t + j);

    V out0;
    V out1;
    if constexpr (forward) {
      out0 = hn::Sub(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Add(hn::Mul(x0, s), hn::Mul(x1, c));
    } else {
      out0 = hn::Add(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Sub(hn::Mul(x1, c), hn::Mul(x0, s));
    }

    store_f32_as_typed(d, out0, x_out, j);
    store_f32_as_typed(d, out1, x_out, j + half_dims);
  }
  return j;
}

template <bool forward>
int rope_traditional_dispatch(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  switch (dtype) {
    case RopeHighwayDType::Float32:
      return rope_traditional<float, forward>(
          static_cast<const float*>(x_in),
          static_cast<float*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::Float16:
      return rope_traditional<float16_t, forward>(
          static_cast<const float16_t*>(x_in),
          static_cast<float16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::BFloat16:
      return rope_traditional<bfloat16_t, forward>(
          static_cast<const bfloat16_t*>(x_in),
          static_cast<bfloat16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
  }
  return 0;
}

template <bool forward>
int rope_non_traditional_dispatch(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  switch (dtype) {
    case RopeHighwayDType::Float32:
      return rope_non_traditional<float, forward>(
          static_cast<const float*>(x_in),
          static_cast<float*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::Float16:
      return rope_non_traditional<float16_t, forward>(
          static_cast<const float16_t*>(x_in),
          static_cast<float16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::BFloat16:
      return rope_non_traditional<bfloat16_t, forward>(
          static_cast<const bfloat16_t*>(x_in),
          static_cast<bfloat16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
  }
  return 0;
}

void RopeTraditionalForward(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_traditional_dispatch<true>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

void RopeTraditionalReverse(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_traditional_dispatch<false>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

void RopeNonTraditionalForward(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_non_traditional_dispatch<true>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

void RopeNonTraditionalReverse(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_non_traditional_dispatch<false>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

} // namespace
} // namespace HWY_NAMESPACE
} // namespace mlx::core::fast
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace mlx::core::fast {

HWY_EXPORT(RopeTraditionalForward);
HWY_EXPORT(RopeTraditionalReverse);
HWY_EXPORT(RopeNonTraditionalForward);
HWY_EXPORT(RopeNonTraditionalReverse);

int rope_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeTraditionalForward)(
      x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int rope_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeTraditionalReverse)(
      x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int rope_non_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeNonTraditionalForward)(
      x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int rope_non_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeNonTraditionalReverse)(
      x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

} // namespace mlx::core::fast
#endif // HWY_ONCE
