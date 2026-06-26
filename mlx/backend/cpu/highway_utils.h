// Copyright © 2026 Apple Inc.

// This header is included once per Highway target by foreach_target.h; do not
// add an include guard.

#include <cstddef>
#include <type_traits>

#include "hwy/highway.h"
#include "mlx/types/half_types.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core::highway {
namespace HWY_NAMESPACE {

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

} // namespace HWY_NAMESPACE
} // namespace mlx::core::highway
HWY_AFTER_NAMESPACE();
