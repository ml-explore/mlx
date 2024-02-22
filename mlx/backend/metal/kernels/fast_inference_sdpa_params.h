//
//  MLXFastInferenceSDPAParams.h
//  mlx
//
//  Copyright (C) 2024 Argmax, Inc.
//

#pragma once

struct MLXFastInferenceSDPAParams {
  // Associated dimensions & transposition information
  const uint B = 1;
  const uint R = 1;
  const uint N_Q_HEADS = 32;
  const uint N_KV_HEADS = 32;
  const bool transpose_q = false;
  const bool transpose_k = false;
  const bool transpose_v = false;
  const uint KV_TILES = 1;
  const float INV_ALPHA = 0.08838834764831843f;
};
