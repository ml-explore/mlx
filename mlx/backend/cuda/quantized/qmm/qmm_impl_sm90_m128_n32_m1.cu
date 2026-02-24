// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/qmm/qmm_impl_sm90.cuh"

using namespace cute;

using TileShapeMN = Shape<_128, _32>;
using ClusterShape = Shape<_1, _1, _1>;

QMM_SM90_GPU(TileShapeMN, ClusterShape)
