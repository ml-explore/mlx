// Copyright © 2025 Apple Inc.

#pragma once

#include <optional>

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

class array;
struct Stream;

array compute_first_token_offset(
    const array& indices,
    int group_count,
    cu::CommandEncoder& encoder);

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

void cutlass_segmented_mm(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int num_segments,
    int M,
    int N,
    const array& a,
    const array& b,
    const array& segments,
    array& out,
    cu::CommandEncoder& encoder);

void cudnn_grouped_mm(
    const array& x,
    const array& w,
    array& out,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
