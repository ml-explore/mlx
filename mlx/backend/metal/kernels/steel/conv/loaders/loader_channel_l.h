// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/utils.h"

#include "mlx/backend/metal/kernels/steel/conv/params.h"

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
struct Conv2DInputBlockLoaderLargeFilter {
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
  const constant ImplicitGemmConv2DParams* gemm_params;

  short weight_h;
  short weight_w;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderLargeFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];
      int ih = read_ih[i] + weight_h * params->kdil[0];
      int iw = read_iw[i] + weight_w * params->kdil[1];

      // Read from input if in bounds
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
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
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
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
struct Conv2DInputBlockLoaderSmallFilter {
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

  using mask_t = short;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;

  short weight_h;
  short weight_w;

  const device T* src[n_rows];

  mask_t mask_h[n_rows];
  mask_t mask_w[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderSmallFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    int read_n[n_rows];
    int read_ih[n_rows];
    int read_iw[n_rows];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      mask_h[i] = 0;
      mask_w[i] = 0;
    }

    for (short kh = 0; kh < params->wS[0]; kh++) {
      short flip_h = params->flip ? params->wS[0] - kh - 1 : kh;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int n = read_n[i];
        int ih = read_ih[i] + flip_h * params->kdil[0];

        bool in_bounds = n < params->N && ih >= 0 && ih < params->iS[0];

        mask_h[i] |= (in_bounds << kh);
      }
    }

    for (short kw = 0; kw < params->wS[1]; kw++) {
      short flip_w = params->flip ? params->wS[1] - kw - 1 : kw;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int iw = read_iw[i] + flip_w * params->kdil[1];

        bool in_bounds = iw >= 0 && iw < params->iS[1];

        mask_w[i] |= (in_bounds << kw);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    mask_t h_mask = mask_t(1) << weight_h;
    mask_t w_mask = mask_t(1) << weight_w;

    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Read from input if in bounds
      if ((mask_h[i] & h_mask) && (mask_w[i] & w_mask)) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
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
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
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
struct Conv2DWeightBlockLoader {
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

  int weight_hw;
  int weight_step;

  const int read_n;
  const bool do_read;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoader(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        weight_hw(0),
        weight_step(params->C / params->groups),
        read_n(offsets.y + bi),
        do_read(read_n + n_rows * TROWS <= gemm_params_->N) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((read_n + i) < params->O) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = src[i * src_ld + j];
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
    if (++weight_hw < (params->wS[1] * params->wS[0])) {
      src += weight_step;
      return;
    }

    weight_hw = 0;

    src += BK - (params->wS[1] * params->wS[0] - 1) * weight_step;
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv3DInputBlockLoaderLargeFilter {
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

  const constant MLXConvParams<3>* params;
  const constant ImplicitGemmConv3DParams* gemm_params;

  short weight_d;
  short weight_h;
  short weight_w;

  short kdil_d;
  short kdil_h;
  short kdil_w;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_id[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv3DInputBlockLoaderLargeFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<3>* params_,
      const constant ImplicitGemmConv3DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_d(0),
        weight_h(0),
        weight_w(0),
        kdil_d(params_->flip ? -params_->kdil[0] : params_->kdil[0]),
        kdil_h(params_->flip ? -params_->kdil[1] : params_->kdil[1]),
        kdil_w(params_->flip ? -params_->kdil[2] : params_->kdil[2]) {
    int out_n_pixels = params->oS[0] * params->oS[1] * params->oS[2];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_ndhw = offsets.y + bi + i * TROWS;
      int n = offset_ndhw / out_n_pixels;
      int dhw = offset_ndhw % out_n_pixels;
      int od = dhw / (params->oS[1] * params->oS[2]);
      int hw = dhw % (params->oS[1] * params->oS[2]);
      int oh = hw / params->oS[2];
      int ow = hw % params->oS[2];

      int id = od * params->str[0] - params->pad[0];
      int ih = oh * params->str[1] - params->pad[1];
      int iw = ow * params->str[2] - params->pad[2];

      read_n[i] = n;

      if (params->flip) {
        read_id[i] = id + (params->wS[0] - 1) * params->kdil[0];
        read_ih[i] = ih + (params->wS[1] - 1) * params->kdil[1];
        read_iw[i] = iw + (params->wS[2] - 1) * params->kdil[2];
      } else {
        read_id[i] = id;
        read_ih[i] = ih;
        read_iw[i] = iw;
      }

      // Adjust for flip
      if (params->flip) {
        id += (params->wS[0] - 1) * params->kdil[0];
        ih += (params->wS[1] - 1) * params->kdil[1];
        iw += (params->wS[2] - 1) * params->kdil[2];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + id * params->in_strides[1] +
          ih * params->in_strides[2] + iw * params->in_strides[3] + bj;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];
      int id = read_id[i] + weight_d * kdil_d;
      int ih = read_ih[i] + weight_h * kdil_h;
      int iw = read_iw[i] + weight_w * kdil_w;

      // Read from input if in bounds
      if ((n < params->N) && (id >= 0 && id < params->iS[0]) &&
          (ih >= 0 && ih < params->iS[1]) && (iw >= 0 && iw < params->iS[2])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
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
    if (++weight_w < params->wS[2]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    if (++weight_d < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_d;
      }

      return;
    }

    weight_d = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
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
struct Conv3DInputBlockLoaderSmallFilter {
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

  using mask_t = short;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<3>* params;
  const constant ImplicitGemmConv3DParams* gemm_params;

  short weight_d;
  short weight_h;
  short weight_w;

  const device T* src[n_rows];

  mask_t mask_d[n_rows];
  mask_t mask_h[n_rows];
  mask_t mask_w[n_rows];

  /* Constructor */
  METAL_FUNC Conv3DInputBlockLoaderSmallFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<3>* params_,
      const constant ImplicitGemmConv3DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_d(0),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1] * params->oS[2];

    int read_n[n_rows];
    int read_id[n_rows];
    int read_ih[n_rows];
    int read_iw[n_rows];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_ndhw = offsets.y + bi + i * TROWS;
      int n = offset_ndhw / out_n_pixels;
      int dhw = offset_ndhw % out_n_pixels;
      int od = dhw / (params->oS[1] * params->oS[2]);
      int hw = dhw % (params->oS[1] * params->oS[2]);
      int oh = hw / params->oS[2];
      int ow = hw % params->oS[2];

      int id = od * params->str[0] - params->pad[0];
      int ih = oh * params->str[1] - params->pad[1];
      int iw = ow * params->str[2] - params->pad[2];

      read_n[i] = n;
      read_id[i] = id;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        id += (params->wS[0] - 1) * params->kdil[0];
        ih += (params->wS[1] - 1) * params->kdil[1];
        iw += (params->wS[2] - 1) * params->kdil[2];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + id * params->in_strides[1] +
          ih * params->in_strides[2] + iw * params->in_strides[3] + bj;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      mask_d[i] = 0;
      mask_h[i] = 0;
      mask_w[i] = 0;
    }

    for (short kd = 0; kd < params->wS[0]; kd++) {
      short flip_d = params->flip ? params->wS[0] - kd - 1 : kd;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int n = read_n[i];
        int id = read_id[i] + flip_d * params->kdil[0];

        bool in_bounds = n < params->N && id >= 0 && id < params->iS[0];

        mask_d[i] |= (in_bounds << kd);
      }
    }

    for (short kh = 0; kh < params->wS[1]; kh++) {
      short flip_h = params->flip ? params->wS[1] - kh - 1 : kh;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int ih = read_ih[i] + flip_h * params->kdil[1];

        bool in_bounds = ih >= 0 && ih < params->iS[1];

        mask_h[i] |= (in_bounds << kh);
      }
    }

    for (short kw = 0; kw < params->wS[2]; kw++) {
      short flip_w = params->flip ? params->wS[2] - kw - 1 : kw;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int iw = read_iw[i] + flip_w * params->kdil[2];

        bool in_bounds = iw >= 0 && iw < params->iS[2];

        mask_w[i] |= (in_bounds << kw);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    mask_t d_mask = mask_t(1) << weight_d;
    mask_t h_mask = mask_t(1) << weight_h;
    mask_t w_mask = mask_t(1) << weight_w;

    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Read from input if in bounds
      if ((mask_d[i] & d_mask) && (mask_h[i] & h_mask) &&
          (mask_w[i] & w_mask)) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
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
    if (++weight_w < params->wS[2]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    if (++weight_d < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_d;
      }

      return;
    }

    weight_d = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
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
struct Conv3DWeightBlockLoader {
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

  const constant MLXConvParams<3>* params;

  int weight_dhw;
  int weight_step;

  const int read_n;
  const bool do_read;

  /* Constructor */
  METAL_FUNC Conv3DWeightBlockLoader(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<3>* params_,
      const constant ImplicitGemmConv3DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        weight_dhw(0),
        weight_step(params->C / params->groups),
        read_n(offsets.y + bi),
        do_read(read_n + n_rows * TROWS <= gemm_params_->N) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((read_n + i) < params->O) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = src[i * src_ld + j];
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
    if (++weight_dhw < (params->wS[0] * params->wS[1] * params->wS[2])) {
      src += weight_step;
      return;
    }

    weight_dhw = 0;

    src +=
        BK - (params->wS[0] * params->wS[1] * params->wS[2] - 1) * weight_step;
  }
};

} // namespace steel
} // namespace mlx
