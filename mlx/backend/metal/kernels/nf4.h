// Copyright © 2025 Apple Inc.
// NormalFloat4 (NF4) lookup table quantization
// Based on QLoRA: https://arxiv.org/abs/2305.14314

#pragma once

// The 16 NF4 quantization levels, derived from the quantiles of N(0,1).
// Each 4-bit index maps to one of these values. Weights are reconstructed
// as: weight = nf4_lut[index] * block_absmax
//
// The values are asymmetric: 8 negative, 1 zero, 7 positive.
// They are normalized to [-1, 1] and placed so each bin captures
// equal probability mass under a standard normal distribution.
constant constexpr float nf4_lut[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};
