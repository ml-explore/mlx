// Copyright © 2023 Apple Inc.

#pragma once

#include <cstdint>
#include <utility>

namespace mlx::core::random {

/** Applies the Threefry 2x32 hash function.
 * This code is based on the Jax counter-based and splittable PRNG
 * https://github.com/google/jax/blob/main/docs/jep/263-prng.md
 *
 * Original Threefry reference:
 * http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 */
std::pair<uint32_t, uint32_t> threefry2x32_hash(
    const std::pair<uint32_t, uint32_t>& key,
    std::pair<uint32_t, uint32_t> count);

} // namespace mlx::core::random
