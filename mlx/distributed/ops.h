// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/distributed/distributed.h"
#include "mlx/mlx_export.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

MLX_API array all_sum(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

MLX_API array all_gather(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice S = {});

MLX_API array send(
    const array& x,
    int dst,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

MLX_API array recv(
    Shape shape,
    Dtype dtype,
    int src,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

MLX_API array recv_like(
    const array& x,
    int src,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

MLX_API array all_max(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

MLX_API array all_min(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

MLX_API array sum_scatter(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::distributed
