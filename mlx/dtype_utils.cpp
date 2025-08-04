// Copyright © 2025 Apple Inc.

#include "mlx/dtype_utils.h"

namespace mlx::core {

const char* dtype_to_string(Dtype arg) {
  switch (arg) {
    case bool_:
      return "bool";
    case int8:
      return "int8";
    case int16:
      return "int16";
    case int32:
      return "int32";
    case int64:
      return "int64";
    case uint8:
      return "uint8";
    case uint16:
      return "uint16";
    case uint32:
      return "uint32";
    case uint64:
      return "uint64";
    case float16:
      return "float16";
    case bfloat16:
      return "bfloat16";
    case float32:
      return "float32";
    case float64:
      return "float64";
    case complex64:
      return "complex64";
    default:
      return "unknown";
  }
}

} // namespace mlx::core
