// Copyright © 2026 Apple Inc.

#pragma once

// helper function to identify the tile, called from gather_mm / gather_qmm
template <int BM>
METAL_FUNC bool gather_mm_row_tile(
    const device int32_t* offsets,
    const int num_groups,
    const int M,
    const int tile,
    const uint simd_lane_id,
    thread int& row,
    thread int& group,
    thread short& rows) {
  int tiles_before = 0;
  // each lane process an expert
  for (int e = 0; e < num_groups; e += 32) {
    const int g = e + simd_lane_id; // shift by lane id
    int start = M;
    int end = M;
    if (g < num_groups) {
      // start of the experts activations
      start = offsets[g];
      // end of the experts activations
      end = g + 1 < num_groups ? offsets[g + 1] : M;
    }
    // number of tiles per expert
    const int n = (end - start + BM - 1) / BM;
    // the total number of tiles up to and including this expert
    const int tiles_through = tiles_before + simd_prefix_inclusive_sum(n);
    const ushort owner_lane = ushort(simd_sum(int(tiles_through <= tile)));
    if (owner_lane < 32) {
      // if true, we found an owner
      group = e + lane;
      // fill the row to start
      row = simd_shuffle(start, lane) +
          (tile - simd_shuffle(tiles_through - n, lane)) * BM;
      // number of valid rows in the tile
      rows = short(min(BM, simd_shuffle(end, lane) - row));
      return true;
    }
    tiles_before = simd_shuffle(tiles_through, 31);
  }
  return false;
}
