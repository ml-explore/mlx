// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

using namespace metal;

[[kernel]] void gather_mm_offsets(
    const device uint32_t* indices [[buffer(0)]],
    device int32_t* offsets [[buffer(1)]],
    const constant int& M [[buffer(2)]],
    uint g [[thread_position_in_grid]]) {
  int lo = 0;
  int hi = M;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (indices[mid] < g) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  offsets[g] = lo;
}
