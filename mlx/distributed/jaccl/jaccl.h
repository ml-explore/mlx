// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>
#include <functional>

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::jaccl {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

/**
 * Byte-level all-gather function used by the JACCL side channel.
 *
 * Args:
 *   src: Pointer to this rank's data of size n_bytes.
 *   dst: Pointer to an output buffer of size size_ * n_bytes. After the call,
 *        dst[rank * n_bytes, (rank+1) * n_bytes] contains the data from rank.
 *   n_bytes: The number of bytes contributed by each rank.
 */
using AllGatherFn = std::function<void(const char*, char*, size_t)>;

/**
 * Factory that produces a per-rank side-channel all-gather function.
 *
 * The factory receives the rank and size of the group and returns the
 * AllGatherFn that will be used for the side channel.
 */
using AllGatherFactory = std::function<AllGatherFn(int, int)>;

MLX_API bool is_available();
MLX_API std::shared_ptr<GroupImpl> init(bool strict = false);
MLX_API std::shared_ptr<GroupImpl> init(bool strict, AllGatherFactory factory);

} // namespace mlx::core::distributed::jaccl
