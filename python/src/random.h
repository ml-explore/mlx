// Copyright © 2026 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>

#include "mlx/array.h"

namespace mx = mlx::core;
namespace nb = nanobind;

// The process-global `mx.random.state` sentinel.
nb::object random_state_sentinel();

// True if `obj` is the RNG state sentinel.
bool is_random_state(nb::handle obj);

// Read/write the calling thread's current PRNG key.
mx::array random_state_key();
void set_random_state_key(const mx::array& key);
