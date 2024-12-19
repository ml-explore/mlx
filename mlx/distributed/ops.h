// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/distributed/distributed.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

array all_sum(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

array all_gather(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice S = {});

array send(
    const array& x,
    int dst,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

array recv(
    Shape shape,
    Dtype dtype,
    int src,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

array recv_like(
    const array& x,
    int src,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::distributed
