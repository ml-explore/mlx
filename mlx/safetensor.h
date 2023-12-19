// Copyright © 2023 Apple Inc.

#pragma once

#include <json.hpp>

#include "mlx/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

using json = nlohmann::json;

namespace mlx::core {

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
} // namespace mlx::core
