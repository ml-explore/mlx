// Copyright © 2024 Apple Inc.

#include "mlx/array.h"
#include "mlx/io/load.h"

namespace mlx::core {

void load(
    array& out,
    size_t offset,
    const std::shared_ptr<io::Reader>& reader,
    bool swap_endianess);

} // namespace mlx::core
