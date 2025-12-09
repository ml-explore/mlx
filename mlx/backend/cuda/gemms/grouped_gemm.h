// Copyright © 2025 Apple Inc.

#pragma once

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

class array;

void cutlass_grouped_gemm_unaligned(
    const array& a,
    const array& b,
    const array& indices,
    int group_count,
    array& out,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
