// Copyright © 2026 Apple Inc.

#include <stdexcept>

#include "python/src/dlpack_consumer.h"

mx::array build_dlpack_metal_array(
    nb::dlpack::dltensor& /*t*/,
    std::shared_ptr<DLPackOwner> /*owner*/) {
  throw std::invalid_argument(
      "[array] MLX was built without Metal support; cannot consume "
      "kDLMetal capsules.");
}
