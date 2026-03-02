// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>
#include <string>

#include "mlx/api.h"
#include "mlx/distributed/distributed.h"
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

MLX_API array all_to_all(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

MLX_API std::pair<array, array> moe_dispatch_exchange(
    const array& tokens,
    const array& expert_indices,
    int num_experts,
    int capacity,
    std::optional<Group> group = std::nullopt,
    bool deterministic = true,
    const std::string& backend = "cpu",
    StreamOrDevice s = {});

MLX_API array moe_combine_exchange(
    const array& expert_outputs,
    const array& route_indices,
    const array& weights,
    const array& original_tokens,
    int num_experts,
    int capacity,
    std::optional<Group> group = std::nullopt,
    bool deterministic = true,
    const std::string& backend = "cpu",
    StreamOrDevice s = {});

} // namespace mlx::core::distributed
