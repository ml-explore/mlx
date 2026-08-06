// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/fft/readwrite.h"

// Compile-only regression for the FFT complex_t storage plumbing. Exercises
// the packed complex_t<T> lane conversions and inverse conjugation for lane
// types the production kernels do not instantiate.
template <typename T>
kernel void fft_complex_storage(
    const device complex_t<T>* in [[buffer(0)]],
    device complex_t<T>* out [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  vec<T, 2> value = in[index];
  value = vec<T, 2>(value.x, -value.y);
  out[index] = value;
}

template [[host_name("fft_complex_storage_half")]] [[kernel]]
decltype(fft_complex_storage<half>) fft_complex_storage<half>;

template [[host_name("fft_complex_storage_bfloat16")]] [[kernel]]
decltype(fft_complex_storage<bfloat16_t>) fft_complex_storage<bfloat16_t>;
