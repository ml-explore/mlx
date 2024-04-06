//
//  scaled_dot_product_attention_params.h
//  mlx

#pragma once

struct MLXFastAttentionParams {
  const int M;
  const int N;
  const int K;

  const int ldq; // ldq == ldo
  const int ldk;
  const int ldv;
  const int lds;
  const int ldo;

  const int tiles_n;
  const int tiles_m;

  const int batch_stride_q;
  const int batch_stride_k;
  const int batch_stride_v;
  const int batch_stride_o;

  const int swizzle_log;
  const int gemm_n_iterations_aligned;
  const int gemm_k_iterations_aligned;
  const int gemm_sv_m_block_iterations;

  const int batch_ndim;
  const float alpha;
};

struct MLXScaledDotProductAttentionParams {
  // Associated dimensions & transposition information
  const uint QUERY_SEQUENCE_LENGTH = 1;
  const uint N_Q_HEADS = 32;
  const uint N_KV_HEADS = 32;
  const uint KV_TILES = 1;
  const float INV_ALPHA = 0.08838834764831843f;
};
