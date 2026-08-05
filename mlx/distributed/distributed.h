// Copyright © 2024 Apple Inc.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

// Forward declaration of the base group implementation.
namespace detail {
class GroupImpl;
}

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

/* Check if a communication backend is available */
MLX_API bool is_available();
MLX_API bool is_available(const std::string& bk);

/**
 * A distributed::Group represents a group of independent mlx processes that
 * can communicate. We must also be able to create sub-groups from a group in
 * order to define more granular communication.
 */
struct MLX_API Group {
  Group(std::shared_ptr<detail::GroupImpl> group) : group_(std::move(group)) {}

  int rank() const;
  int size() const;

  /**
   * Split the group according to the provided color. Namely processes that use
   * the same color will go to the same group.
   *
   * The key defines the rank of the processes in the new group. The smaller
   * the key the smaller the rank. If the provided key is negative, then the
   * rank in the current group is used.
   */
  Group split(int color, int key = -1) const;

  const std::shared_ptr<detail::GroupImpl>& raw_group() const {
    return group_;
  }

 private:
  std::shared_ptr<detail::GroupImpl> group_{nullptr};
};

/**
 * Initialize the distributed backend and return the group containing all
 * discoverable processes.
 *
 * If strict is true then throw an error if we couldn't initialize the
 * distributed subsystem. Otherwise simply return a singleton group which will
 * render communication operations as no-op.
 */
MLX_API Group init(bool strict = false, const std::string& bk = "any");

/**
 * Initialize the distributed backend using a custom JACCL side channel.
 *
 * This behaves like init(strict, bk) but requires bk to be "jaccl".
 */
MLX_API Group
init(bool strict, const std::string& bk, AllGatherFactory factory);

/** Clear the distributed backend cache. */
MLX_API void clear_backends();

} // namespace mlx::core::distributed
