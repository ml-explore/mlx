// Copyright © 2026 Apple Inc.

// Normally this file is compiled directly and Highway emits its runtime
// dispatch targets. Native MSVC builds compile this file once per target with
// MLX_HIGHWAY_MANUAL_TARGET and MLX_HIGHWAY_TARGET_SUFFIX so the same kernels
// are emitted as one manually suffixed specialization.

#include <algorithm>
#include <cstddef>
#include <type_traits>

#if !defined(MLX_HIGHWAY_MANUAL_TARGET)
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/conv_highway.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep
#endif

#include "hwy/highway.h"
#include "mlx/backend/cpu/conv_highway.h"
#include "mlx/backend/cpu/highway_utils.h"
#include "mlx/backend/cpu/threading/common.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core::fast {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;
namespace hu = mlx::core::highway::HWY_NAMESPACE;

template <typename T, class DF>
void load_interleaved4_typed_as_f32(
    DF df,
    const T* HWY_RESTRICT ptr,
    hn::Vec<DF>& w0,
    hn::Vec<DF>& w1,
    hn::Vec<DF>& w2,
    hn::Vec<DF>& w3) {
  if constexpr (std::is_same_v<T, float>) {
    hn::LoadInterleaved4(df, ptr, w0, w1, w2, w3);
  } else if constexpr (std::is_same_v<T, float16_t>) {
    const hn::Rebind<hwy::float16_t, DF> df16;
    hn::Vec<decltype(df16)> w0h;
    hn::Vec<decltype(df16)> w1h;
    hn::Vec<decltype(df16)> w2h;
    hn::Vec<decltype(df16)> w3h;
    hn::LoadInterleaved4(
        df16, reinterpret_cast<const hwy::float16_t*>(ptr), w0h, w1h, w2h, w3h);
    w0 = hn::PromoteTo(df, w0h);
    w1 = hn::PromoteTo(df, w1h);
    w2 = hn::PromoteTo(df, w2h);
    w3 = hn::PromoteTo(df, w3h);
  } else {
#if HWY_TARGET == HWY_SCALAR
    const hn::Rebind<hwy::bfloat16_t, DF> dbf16;
#else
    const hn::Repartition<hwy::bfloat16_t, DF> dbf16;
#endif
    const hn::Half<decltype(dbf16)> dbf16_half;
    hn::Vec<decltype(dbf16_half)> w0h;
    hn::Vec<decltype(dbf16_half)> w1h;
    hn::Vec<decltype(dbf16_half)> w2h;
    hn::Vec<decltype(dbf16_half)> w3h;
    hn::LoadInterleaved4(
        dbf16_half,
        reinterpret_cast<const hwy::bfloat16_t*>(ptr),
        w0h,
        w1h,
        w2h,
        w3h);
    w0 = hn::PromoteTo(df, w0h);
    w1 = hn::PromoteTo(df, w1h);
    w2 = hn::PromoteTo(df, w2h);
    w3 = hn::PromoteTo(df, w3h);
  }
}

template <typename T>
void depthwise_conv1d_kernel4_typed(
    T* HWY_RESTRICT out,
    const T* HWY_RESTRICT in,
    const T* HWY_RESTRICT wt,
    int iH,
    int C,
    int oH,
    size_t row_begin,
    size_t row_end) {
  const hn::ScalableTag<float> df;
  const size_t lanes = hn::Lanes(df);

  for (size_t row = row_begin; row < row_end; ++row) {
    const int n = static_cast<int>(row / oH);
    const int oh = static_cast<int>(row - static_cast<size_t>(n) * oH);
    const T* in_row = in + (static_cast<size_t>(n) * iH + oh) * C;
    T* out_row = out + row * C;

    size_t c = 0;
    for (; c + lanes <= static_cast<size_t>(C); c += lanes) {
      const auto x0 = hu::load_typed_as_f32(df, in_row, c);
      const auto x1 = hu::load_typed_as_f32(df, in_row + C, c);
      const auto x2 = hu::load_typed_as_f32(df, in_row + 2 * C, c);
      const auto x3 = hu::load_typed_as_f32(df, in_row + 3 * C, c);
      hn::Vec<decltype(df)> w0;
      hn::Vec<decltype(df)> w1;
      hn::Vec<decltype(df)> w2;
      hn::Vec<decltype(df)> w3;
      load_interleaved4_typed_as_f32(df, wt + c * 4, w0, w1, w2, w3);

      auto acc = hn::Mul(x0, w0);
      acc = hn::MulAdd(x1, w1, acc);
      acc = hn::MulAdd(x2, w2, acc);
      acc = hn::MulAdd(x3, w3, acc);
      hu::store_f32_as_typed(df, acc, out_row, c);
    }

    for (; c < static_cast<size_t>(C); ++c) {
      const T* wt_row = wt + c * 4;
      const float acc =
          static_cast<float>(in_row[c]) * static_cast<float>(wt_row[0]) +
          static_cast<float>(in_row[C + c]) * static_cast<float>(wt_row[1]) +
          static_cast<float>(in_row[2 * C + c]) *
              static_cast<float>(wt_row[2]) +
          static_cast<float>(in_row[3 * C + c]) * static_cast<float>(wt_row[3]);
      out_row[c] = static_cast<T>(acc);
    }
  }
}

template <typename T>
void depthwise_conv1d_kernel4_impl(
    void* out,
    const void* in,
    const void* wt,
    int N,
    int iH,
    int C,
    int oH) {
  auto& pool = cpu::ThreadPool::instance();
  const size_t rows = static_cast<size_t>(N) * oH;
  const size_t work = rows * C;
  const int n_threads = cpu::effective_threads(work, pool.max_threads());
  cpu::parallel_for_range(n_threads, rows, [&](size_t begin, size_t end) {
    depthwise_conv1d_kernel4_typed(
        static_cast<T*>(out),
        static_cast<const T*>(in),
        static_cast<const T*>(wt),
        iH,
        C,
        oH,
        begin,
        end);
  });
}

} // namespace

void DepthwiseConv1DKernel4(
    void* out,
    const void* in,
    const void* wt,
    ConvHighwayDType dtype,
    int N,
    int iH,
    int C,
    int oH) {
  switch (dtype) {
    case ConvHighwayDType::Float32:
      return depthwise_conv1d_kernel4_impl<float>(out, in, wt, N, iH, C, oH);
    case ConvHighwayDType::Float16:
      return depthwise_conv1d_kernel4_impl<float16_t>(
          out, in, wt, N, iH, C, oH);
    case ConvHighwayDType::BFloat16:
      return depthwise_conv1d_kernel4_impl<bfloat16_t>(
          out, in, wt, N, iH, C, oH);
  }
}

} // namespace HWY_NAMESPACE
} // namespace mlx::core::fast
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace mlx::core::fast {

#if defined(MLX_HIGHWAY_MANUAL_TARGET)

#ifndef MLX_HIGHWAY_TARGET_SUFFIX
#error "MLX_HIGHWAY_TARGET_SUFFIX must be defined for manual Highway targets"
#endif

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)
#define MLX_HIGHWAY_TARGET_FUNC(name) \
  MLX_HIGHWAY_CONCAT(name, MLX_HIGHWAY_TARGET_SUFFIX)

void MLX_HIGHWAY_TARGET_FUNC(depthwise_conv1d_kernel4_highway)(
    void* out,
    const void* in,
    const void* wt,
    ConvHighwayDType dtype,
    int N,
    int iH,
    int C,
    int oH) {
  HWY_STATIC_DISPATCH(DepthwiseConv1DKernel4)
  (out, in, wt, dtype, N, iH, C, oH);
}

#undef MLX_HIGHWAY_TARGET_FUNC
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

#else

HWY_EXPORT(DepthwiseConv1DKernel4);

void depthwise_conv1d_kernel4_highway(
    void* out,
    const void* in,
    const void* wt,
    ConvHighwayDType dtype,
    int N,
    int iH,
    int C,
    int oH) {
  HWY_DYNAMIC_DISPATCH(DepthwiseConv1DKernel4)
  (out, in, wt, dtype, N, iH, C, oH);
}

#endif

} // namespace mlx::core::fast
#endif // HWY_ONCE
