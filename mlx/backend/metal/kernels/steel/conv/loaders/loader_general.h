// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"

///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderGeneral {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant Conv2DGeneralJumpParams* jump_params;

  const short base_wh;
  const short base_ww;

  short weight_h;
  short weight_w;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderGeneral(
      const device T* src_,
      threadgroup T* dst_,
      const int4 offsets,
      const constant MLXConvParams<2>* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_,
      const short base_ww_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / jump_params->adj_out_hw;
      int hw = offset_nhw % jump_params->adj_out_hw;
      int oh =
          (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h + offsets.z;
      int ow =
          (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w + offsets.w;

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + bj;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];

      int h_flip = params->flip ? params->wS[0] - weight_h - 1 : weight_h;
      int w_flip = params->flip ? params->wS[1] - weight_w - 1 : weight_w;

      int ih_dil = read_ih[i] + h_flip * params->kdil[0];
      int iw_dil = read_iw[i] + w_flip * params->kdil[1];

      int ih = ih_dil / params->idil[0];
      int iw = iw_dil / params->idil[1];

      size_t offset = ih * params->in_strides[1] + iw * params->in_strides[2];

      // Read from input if in bounds
      if ((n < params->N) && (ih_dil >= 0 && ih < params->iS[0]) &&
          (iw_dil >= 0 && iw < params->iS[1])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = (src[i])[offset + j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) {
      return;
    }

    weight_w = base_ww;

    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) {
      return;
    }

    weight_h = base_wh;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += BK;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DWeightBlockLoaderGeneral {
  // Destination dimensions
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>* params;
  const constant Conv2DGeneralJumpParams* jump_params;

  const short base_wh;
  const short base_ww;

  short weight_h;
  short weight_w;

  const int start_row;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoaderGeneral(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_,
      const short base_ww_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_),
        start_row(offsets.y + bi) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    const device T* curr_src = src + weight_h * params->wt_strides[1] +
        weight_w * params->wt_strides[2];

    if ((start_row + BN <= params->O)) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((start_row + i) < params->O) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) {
      return;
    }

    weight_w = base_ww;

    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) {
      return;
    }

    weight_h = base_wh;

    src += BK;
  }
};

} // namespace steel
} // namespace mlx
