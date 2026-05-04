// Copyright © 2026 Apple Inc.

#pragma once

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

class array;

void cutlass_gather_mm(
    bool a_transposed,
    bool b_transposed,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
