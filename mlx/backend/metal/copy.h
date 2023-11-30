// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/copy.h"
#include "mlx/stream.h"

namespace mlx::core {

void copy_gpu(const array& src, array& out, CopyType ctype, const Stream& s);
void copy_gpu(const array& src, array& out, CopyType ctype);
void copy_gpu_inplace(
    const array& src,
    array& out,
    CopyType ctype,
    const Stream& s);

} // namespace mlx::core
