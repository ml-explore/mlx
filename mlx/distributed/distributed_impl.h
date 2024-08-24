// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::detail {

/* Return the communication stream. */
Stream communication_stream();

/* Perform an all reduce sum operation */
void all_sum(Group group, const array& input, array& output);

/* Perform an all reduce sum operation */
void all_gather(Group group, const array& input, array& output);

} // namespace mlx::core::distributed::detail
