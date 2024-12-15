// Copyright © 2023-2024 Apple Inc.
#pragma once

#include "mlx/io.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

extern "C" {
#include <gguflib.h>
}

namespace mlx::core {

Shape get_shape(const gguf_tensor& tensor);
void gguf_load_quantized(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor);

} // namespace mlx::core
