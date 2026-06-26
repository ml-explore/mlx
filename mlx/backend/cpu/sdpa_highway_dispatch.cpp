// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/sdpa_highway.h"

#include "hwy/highway.h"

namespace mlx::core::fast {

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)

using SdpaFn = void (*)(
    void*,
    const void*,
    const void*,
    const void*,
    SdpaHighwayDType,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    bool,
    const void*,
    bool,
    const void*,
    bool);

#define MLX_DECLARE_SDPA_TARGET(suffix)          \
  void MLX_HIGHWAY_CONCAT(sdpa_highway, suffix)( \
      void*,                                     \
      const void*,                               \
      const void*,                               \
      const void*,                               \
      SdpaHighwayDType,                          \
      int,                                       \
      int,                                       \
      int,                                       \
      int,                                       \
      int,                                       \
      int,                                       \
      float,                                     \
      bool,                                      \
      const void*,                               \
      bool,                                      \
      const void*,                               \
      bool)

MLX_DECLARE_SDPA_TARGET(_avx2);
MLX_DECLARE_SDPA_TARGET(_sse4);
MLX_DECLARE_SDPA_TARGET(_ssse3);
MLX_DECLARE_SDPA_TARGET(_sse2);

#undef MLX_DECLARE_SDPA_TARGET

namespace {

struct SdpaHighwayDispatch {
  SdpaFn sdpa;
};

#define MLX_SDPA_DISPATCH(suffix)            \
  SdpaHighwayDispatch {                      \
    MLX_HIGHWAY_CONCAT(sdpa_highway, suffix) \
  }

const SdpaHighwayDispatch& sdpa_dispatch() {
  static const SdpaHighwayDispatch dispatch = [] {
    const int64_t targets = hwy::SupportedTargets();
    if (targets & HWY_AVX2) {
      return MLX_SDPA_DISPATCH(_avx2);
    }
    if (targets & HWY_SSE4) {
      return MLX_SDPA_DISPATCH(_sse4);
    }
    if (targets & HWY_SSSE3) {
      return MLX_SDPA_DISPATCH(_ssse3);
    }
    return MLX_SDPA_DISPATCH(_sse2);
  }();
  return dispatch;
}

#undef MLX_SDPA_DISPATCH
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

} // namespace

void sdpa_highway(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    SdpaHighwayDType dtype,
    int B,
    int n_q_heads,
    int n_kv_heads,
    int M,
    int seq_len,
    int head_dim,
    float scale,
    bool do_causal,
    const void* mask,
    bool has_mask,
    const void* sinks,
    bool has_sinks) {
  sdpa_dispatch().sdpa(
      output,
      queries,
      keys,
      values,
      dtype,
      B,
      n_q_heads,
      n_kv_heads,
      M,
      seq_len,
      head_dim,
      scale,
      do_causal,
      mask,
      has_mask,
      sinks,
      has_sinks);
}

} // namespace mlx::core::fast
