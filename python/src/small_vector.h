// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/small_vector.h"

#include <nanobind/stl/detail/nb_list.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename Type, size_t Size, typename Alloc>
struct type_caster<mlx::core::SmallVector<Type, Size, Alloc>>
    : list_caster<mlx::core::SmallVector<Type, Size, Alloc>, Type> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
