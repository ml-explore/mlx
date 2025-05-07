#pragma once

#include <mutex>

namespace mlx::core::gpu {
    extern std::mutex metal_operation_mutex;
}