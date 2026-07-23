// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>

#include "jaccl/types.h"

namespace jaccl {

// Each reduction op has an in place form out[i] OP= in[i] and an out of place
// form out[i] = a[i] OP b[i]. The out of place pointers are __restrict, so
// callers must only use it when a, b and output are distinct buffers.

template <typename T>
struct SumOp {
  void operator()(const T* input, T* output, size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = output[i] + input[i];
    }
  }
  void operator()(
      const T* __restrict a,
      const T* __restrict b,
      T* __restrict output,
      size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = a[i] + b[i];
    }
  }
};

template <typename T>
struct MaxOp {
  void operator()(const T* input, T* output, size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = (output[i] > input[i]) ? output[i] : input[i];
    }
  }
  void operator()(
      const T* __restrict a,
      const T* __restrict b,
      T* __restrict output,
      size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
  }
};

template <typename T>
struct MinOp {
  void operator()(const T* input, T* output, size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = (output[i] < input[i]) ? output[i] : input[i];
    }
  }
  void operator()(
      const T* __restrict a,
      const T* __restrict b,
      T* __restrict output,
      size_t N) const {
    for (size_t i = 0; i < N; i++) {
      output[i] = (a[i] < b[i]) ? a[i] : b[i];
    }
  }
};

//
// The last piece of the puzzle to use the native bf16 while compiling a single
// binary for all Macs is to compile these functions with
// target("arch=armv8.6-a").
//
// Now we can simply check in runtime and call them only when they are
// supported.
//

#if defined(__aarch64__)

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_sum(const void* input, void* output, size_t N) {
  auto in = reinterpret_cast<const __bf16*>(input);
  auto out = reinterpret_cast<__bf16*>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = out[i] + in[i];
  }
}

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_max(const void* input, void* output, size_t N) {
  auto in = reinterpret_cast<const __bf16*>(input);
  auto out = reinterpret_cast<__bf16*>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = (out[i] > in[i]) ? out[i] : in[i];
  }
}

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_min(const void* input, void* output, size_t N) {
  auto in = reinterpret_cast<const __bf16*>(input);
  auto out = reinterpret_cast<__bf16*>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = (out[i] < in[i]) ? out[i] : in[i];
  }
}

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_sum(const void* a, const void* b, void* output, size_t N) {
  auto pa = reinterpret_cast<const __bf16* __restrict>(a);
  auto pb = reinterpret_cast<const __bf16* __restrict>(b);
  auto out = reinterpret_cast<__bf16* __restrict>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = pa[i] + pb[i];
  }
}

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_max(const void* a, const void* b, void* output, size_t N) {
  auto pa = reinterpret_cast<const __bf16* __restrict>(a);
  auto pb = reinterpret_cast<const __bf16* __restrict>(b);
  auto out = reinterpret_cast<__bf16* __restrict>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = (pa[i] > pb[i]) ? pa[i] : pb[i];
  }
}

__attribute__((target("arch=armv8.6-a"))) inline void
native_bf16_min(const void* a, const void* b, void* output, size_t N) {
  auto pa = reinterpret_cast<const __bf16* __restrict>(a);
  auto pb = reinterpret_cast<const __bf16* __restrict>(b);
  auto out = reinterpret_cast<__bf16* __restrict>(output);
  for (size_t i = 0; i < N; i++) {
    out[i] = (pa[i] < pb[i]) ? pa[i] : pb[i];
  }
}

template <>
struct SumOp<bfloat16_t> {
  void operator()(const bfloat16_t* input, bfloat16_t* output, size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_sum(input, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = output[i] + input[i];
      }
    }
  }
  void operator()(
      const bfloat16_t* __restrict a,
      const bfloat16_t* __restrict b,
      bfloat16_t* __restrict output,
      size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_sum(a, b, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = a[i] + b[i];
      }
    }
  }
};

template <>
struct MaxOp<bfloat16_t> {
  void operator()(const bfloat16_t* input, bfloat16_t* output, size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_max(input, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = (output[i] > input[i]) ? output[i] : input[i];
      }
    }
  }
  void operator()(
      const bfloat16_t* __restrict a,
      const bfloat16_t* __restrict b,
      bfloat16_t* __restrict output,
      size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_max(a, b, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = (a[i] > b[i]) ? a[i] : b[i];
      }
    }
  }
};

template <>
struct MinOp<bfloat16_t> {
  void operator()(const bfloat16_t* input, bfloat16_t* output, size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_min(input, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = (output[i] < input[i]) ? output[i] : input[i];
      }
    }
  }
  void operator()(
      const bfloat16_t* __restrict a,
      const bfloat16_t* __restrict b,
      bfloat16_t* __restrict output,
      size_t N) const {
    if (has_native_bf16_support()) {
      native_bf16_min(a, b, output, N);
    } else {
      for (size_t i = 0; i < N; i++) {
        output[i] = (a[i] < b[i]) ? a[i] : b[i];
      }
    }
  }
};

#endif // defined(__aarch64__)

} // namespace jaccl
