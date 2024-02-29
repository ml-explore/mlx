// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

bool is_available();

/* Get the actively used memory in bytes.
 *
 * Note, this will not always match memory use reported by the system because
 * it does not include cached memory buffers.
 * */
size_t get_active_memory();

/* Get the peak amount of used memory in bytes.
 *
 * The maximum memory used is recorded from the beginning of the program
 * execution.
 * */
size_t get_peak_memory();

/* Set the memory limit.
 * Calls to malloc will wait on scheduled tasks if the limit is exceeded.  If
 * there are no more scheduled tasks an error will be raised if relaxed
 * is false or memory will be allocated (including the potential for
 * swap) if relaxed is true.
 *
 * The memory limit defaults to 1.5 times the maximum recommended working set
 * size reported by the device.
 *
 * Returns the previous memory limit.
 * */
size_t set_memory_limit(size_t limit, bool relaxed = true);

/* Set the garbage collection limit.
 * If using more than the given limit, free memory will be reclaimed
 * from the garbage collector on allocation. To disable the garbage collector,
 * set the limit to 0.
 *
 * The gc limit defaults to .95 times the maximum recommended working set
 * size reported by the device.
 *
 * Returns the previous gc limit.
 * */
size_t set_gc_limit(size_t limit);

void new_stream(Stream stream);
std::shared_ptr<void> new_scoped_memory_pool();

std::function<void()> make_task(
    array& arr,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p);

} // namespace mlx::core::metal
