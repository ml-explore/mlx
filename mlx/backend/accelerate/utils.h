// Copyright © 2023 Apple Inc.

#pragma once

#include <vecLib/BNNS/bnns.h>
#include "mlx/dtype.h"

namespace mlx::core {

BNNSDataType to_bnns_dtype(Dtype mlx_dtype) {
  uint32_t size_bits = size_of(mlx_dtype) * 8;
  switch (kindof(mlx_dtype)) {
    case Dtype::Kind::b:
      return BNNSDataTypeBoolean;
    case Dtype::Kind::u:
      return BNNSDataType(BNNSDataTypeUIntBit | size_bits);
    case Dtype::Kind::i:
      return BNNSDataType(BNNSDataTypeIntBit | size_bits);
    case Dtype::Kind::f:
      return BNNSDataType(BNNSDataTypeFloatBit | size_bits);
    case Dtype::Kind::V:
      return BNNSDataTypeBFloat16;
    case Dtype::Kind::c:
      throw std::invalid_argument("BNNS does not support complex types");
  }
}

} // namespace mlx::core