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

std::vector<int> get_shape(const gguf_tensor& tensor);
void extract_q4_0_data(
    std::unordered_map<std::string, array>* a,
    const gguf_tensor& tensor);
void extract_q4_1_data(
    std::unordered_map<std::string, array>* a,
    const gguf_tensor& tensor);
void extract_q8_0_data(
    std::unordered_map<std::string, array>* a,
    const gguf_tensor& tensor);

} // namespace mlx::core
