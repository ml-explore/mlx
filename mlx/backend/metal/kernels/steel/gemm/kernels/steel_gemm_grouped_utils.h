// Copyright © 2026 Apple Inc.

#pragma once

template <int BM>
METAL_FUNC bool grouped_mm_row_tile(
    const device int32_t* offsets,
    const int num_groups,
    const int M,
    const int tile,
    const uint simd_lane_id,
    thread int& row,
    thread int& group,
    thread short& rows) {
  int tiles_before = 0;
  for (int e = 0; e < num_groups; e += 32) {
    const int g = e + simd_lane_id; // each lane reads 1 experts offsets 
    int start = M;
    int end = M;
    if (g < num_groups) {
      start = offsets[g]; // start of the expert's tokens
      end = g + 1 < num_groups ? offsets[g + 1] : M; // end of the expert's tokens
    } 
    const int n = (end - start + BM - 1) / BM; // number of tiles per expert 
    const int tiles_through = tiles_before + simd_prefix_inclusive_sum(n); // 
    const ushort lane = ushort(simd_sum(int(tiles_through <= tile)));
    if (lane < 32) {
      group = e + lane;
      row = simd_shuffle(start, lane) +
          (tile - simd_shuffle(tiles_through - n, lane)) * BM;
      rows = short(min(BM, simd_shuffle(end, lane) - row));
      return true;
    }
    tiles_before = simd_shuffle(tiles_through, 31);
  }
  return false;
}
