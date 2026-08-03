// Copyright © 2026 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>

#include "mlx/array.h"

namespace mx = mlx::core;
namespace nb = nanobind;

// The process-global `mx.random.state` sentinel.
nb::object random_state_sentinel();

// Read/write the calling thread's current PRNG key.
mx::array random_state_key();
void set_random_state_key(const mx::array& key);
