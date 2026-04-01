// Copyright © 2026 Apple Inc.
#pragma once

/**
 * @file apple_silicon_optimizations.h
 * 
 * Apple Silicon M5 Max Specific Optimizations
 * ============================================
 * 
 * This file documents the optimizations applied for Apple Silicon M5 Max and other Max chips.
 * 
 * Key Optimizations:
 * ------------------
 * 
 * 1. GEMM Parameters for 's' (Max) Architecture
 *    - Added specific parameter selection for Max chips in matmul.cpp
 *    - Larger block sizes (64x64) for large matrices
 *    - Better thread group configurations (32x2x2 or 32x4x1)
 *    - Increased buffer sizes and operation counts
 * 
 * 2. Device Buffer Optimization (device.cpp)
 *    - M5 Max (arch_gen >= 25): 70 ops, 70 MB buffer
 *    - Other Max chips (arch_gen < 25): 60 ops, 60 MB buffer
 *    - Previous Max chips: 50 ops, 50 MB buffer
 * 
 * 3. Device Info Detection (device_info.cpp)
 *    - Added "is_max_chip" flag to detect Max architecture
 *    - Device name parsing for "Max" substring
 * 
 * Architecture Suffixes:
 * ----------------------
 * - 'p': Phone chips (M1, M2, etc. base variants)
 * - 'g': GPU base/pro chips (M1 Pro, M2 Pro, etc.)
 * - 's': Max chips (M1 Max, M2 Max, M3 Max, M4 Max, M5 Max)
 * - 'd': Ultra chips (M1 Ultra, M2 Ultra, etc.)
 * 
 * M5 Max Characteristics:
 * -----------------------
 * - Architecture suffix: 's' (Max)
 * - Expected arch_gen: 25 or higher
 * - High memory bandwidth (improved over previous generations)
 * - Enhanced compute units
 * - Optimized unified memory architecture
 */

#include <string>
#include <unordered_map>

namespace mlx::core::metal {

/**
 * Check if the device is a Max chip
 */
inline bool is_max_device(const std::string& device_name) {
    return device_name.find("Max") != std::string::npos;
}

/**
 * Check if the device is M5 Max specifically
 */
inline bool is_m5_max(int arch_gen, const std::string& device_name) {
    return arch_gen >= 25 && is_max_device(device_name);
}

/**
 * Get optimal buffer size for Apple Silicon devices
 */
inline std::tuple<int, int> get_optimal_buffer_params(int arch_gen, const std::string& device_name) {
    // Default to medium device
    int max_ops = 40;
    int max_mb = 40;

    if (is_m5_max(arch_gen, device_name)) {
        // M5 Max: Best performance
        max_ops = 70;
        max_mb = 70;
    } else if (is_max_device(device_name)) {
        // Other Max chips
        max_ops = 60;
        max_mb = 60;
    }

    return std::make_tuple(max_ops, max_mb);
}

} // namespace mlx::core::metal
