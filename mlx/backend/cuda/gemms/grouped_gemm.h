// Copyright © 2025 Apple Inc.

#pragma once

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

class array;

void cutlass_grouped_gemm_unaligned(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int group_count,
    const array& a,
    const array& b,
    const array& indices,
    array& out,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
