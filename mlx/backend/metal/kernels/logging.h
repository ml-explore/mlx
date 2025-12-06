// Copyright © 2025 Apple Inc.

#pragma once

#ifdef MLX_METAL_LOG_ENABLED
#include <metal_logging>

#define MLX_METAL_KERNEL_LOG_DEBUG(fmt, ...) \
  os_log_default.log_debugg(fmt, ##__VA_ARGS__)
#define MLX_METAL_KERNEL_LOG_INFO(fmt, ...) \
  os_log_default.log_info(fmt, ##__VA_ARGS__)
#define MLX_METAL_KERNEL_LOG_ERROR(fmt, ...) \
  os_log_default.log_error(fmt, ##__VA_ARGS__)
#define MLX_METAL_KERNEL_LOG_FAULT(fmt, ...) \
  os_log_default.log_fault(fmt, ##__VA_ARGS__)
#else
#define MLX_METAL_KERNEL_LOG(...)
#define MLX_METAL_KERNEL_LOG_ERROR(...)
#define MLX_METAL_KERNEL_LOG_FAULT(...)
#endif
