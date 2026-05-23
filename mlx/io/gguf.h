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

constexpr const char* kKQuantTypesKey = "__kquant_types__";

Shape get_shape(const gguf_tensor& tensor);

const KQuantCodec* gguf_type_to_kquant_codec(uint32_t gguf_type);

void gguf_load_kquant(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor,
    const KQuantCodec& codec,
    std::vector<std::string>& kquant_entries);

} // namespace mlx::core
