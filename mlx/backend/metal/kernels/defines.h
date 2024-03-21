// Copyright © 2023 Apple Inc.

#pragma once

#ifdef __METAL__
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_BINARY_SPECIALIZED_DIMS = 5;
static MTL_CONST constexpr int MAX_COPY_SPECIALIZED_DIMS = 5;
static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 16;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int SOFTMAX_LOOPED_LIMIT = 4096;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;
