// Copyright © 2025 Apple Inc.

#pragma once

#include <cstdlib>
#include <utility>
#include <vector>

#include "mlx/api.h"

namespace mlx::core {

/* Get the actively used memory in bytes.
 *
 * Note, this will not always match memory use reported by the system because
 * it does not include cached memory buffers.
 * */
MLX_API size_t get_active_memory();

/* Get the peak amount of used memory in bytes.
 *
 * The maximum memory used recorded from the beginning of the program
 * execution or since the last call to reset_peak_memory.
 * */
MLX_API size_t get_peak_memory();

/* Reset the peak memory to zero.
 * */
MLX_API void reset_peak_memory();

/* Get the cache size in bytes.
 *
 * The cache includes memory not currently used that has not been returned
 * to the system allocator.
 * */
MLX_API size_t get_cache_memory();

/* Get the number of live GPU buffer objects (active + cached).
 *
 * On Metal this counts the MTL::Buffer objects created through the allocator
 * that have not yet been released. This count — not bytes — is what is
 * checked against the per-process resource limit (~499k on Apple silicon)
 * in the "[metal::malloc] Resource limit exceeded" error. Returns 0 on
 * backends without a handle limit.
 * */
MLX_API size_t get_active_buffer_count();

/* Get the number of buffer objects currently held by the buffer cache.
 *
 * These are included in get_active_buffer_count().
 * */
MLX_API size_t get_cache_buffer_count();

/* Get a histogram of live GPU buffer objects bucketed by power-of-two
 * size class.
 *
 * Returns (size_class_upper_bound_bytes, count) pairs sorted by size class.
 * The counts sum to get_active_buffer_count().
 * */
MLX_API std::vector<std::pair<size_t, size_t>> get_buffer_histogram();

/* Set the memory limit.
 * The memory limit is a guideline for the maximum amount of memory to use
 * during graph evaluation. If the memory limit is exceeded and there is no
 * more RAM (including swap when available) allocations will result in an
 * exception.
 *
 * When Metal is available the memory limit defaults to 1.5 times the maximum
 * recommended working set size reported by the device.
 *
 * Returns the previous memory limit.
 * */
MLX_API size_t set_memory_limit(size_t limit);

/* Get the current memory limit. */
MLX_API size_t get_memory_limit();

/* Set the cache limit.
 * If using more than the given limit, free memory will be reclaimed
 * from the cache on the next allocation. To disable the cache,
 * set the limit to 0.
 *
 * The cache limit defaults to the memory limit.
 *
 * Returns the previous cache limit.
 * */
MLX_API size_t set_cache_limit(size_t limit);

/* Clear the memory cache. */
MLX_API void clear_cache();

/* Set the wired size limit.
 *
 * Note, this function is only useful when using the Metal backend with
 * macOS 15.0 or higher.
 *
 * The wired limit is the total size in bytes of memory that will be kept
 * resident. The default value is ``0``.
 *
 * Setting a wired limit larger than system wired limit is an error.
 *
 * Returns the previous wired limit.
 * */
MLX_API size_t set_wired_limit(size_t limit);

} // namespace mlx::core
