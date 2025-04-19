// Copyright © 2025 Apple Inc.

#include "mlx/dtype_utils.h"

namespace mlx::core {

const char* dtype_to_string(Dtype arg) {
  if (arg == bool_) {
    return "bool";
  }
#define SPECIALIZE_DtypeToString(CPP_TYPE, DTYPE) \
  if (DTYPE == arg) {                             \
    return #DTYPE;                                \
  }
  MLX_FORALL_DTYPES(SPECIALIZE_DtypeToString)
#undef SPECIALIZE_DtypeToString
  return "(unknown)";
}

} // namespace mlx::core
