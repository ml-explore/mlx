// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::cpu {

void set_error_event(Stream s, Event event);
void clear_error_event(Stream s);
void check_error_event(Stream s, Event event);
void new_stream(Stream s);
void new_thread_unsafe_stream(Stream s);
void eval(array& arr);
void clear_streams();

} // namespace mlx::core::cpu
