// Copyright © 2025 Apple Inc.

#pragma once

#include <cstdlib>

namespace mlx::core {

/* Get the actively used memory in bytes.
 *
 * Note, this will not always match memory use reported by the system because
 * it does not include cached memory buffers.
 * */
size_t get_active_memory();

/* Get the peak amount of used memory in bytes.
 *
 * The maximum memory used recorded from the beginning of the program
 * execution or since the last call to reset_peak_memory.
 * */
size_t get_peak_memory();

/* Reset the peak memory to zero.
 * */
void reset_peak_memory();

/* Get the cache size in bytes.
 *
 * The cache includes memory not currently used that has not been returned
 * to the system allocator.
 * */
size_t get_cache_memory();

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
size_t set_memory_limit(size_t limit);

/* Get the current memory limit. */
size_t get_memory_limit();

/* Set the cache limit.
 * If using more than the given limit, free memory will be reclaimed
 * from the cache on the next allocation. To disable the cache,
 * set the limit to 0.
 *
 * The cache limit defaults to the memory limit.
 *
 * Returns the previous cache limit.
 * */
size_t set_cache_limit(size_t limit);

/* Clear the memory cache. */
void clear_cache();

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
size_t set_wired_limit(size_t limit);

} // namespace mlx::core
