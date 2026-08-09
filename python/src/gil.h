// Copyright © 2026 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// True once CPython has begun finalizing. Py_IsInitialized() is not a
// substitute; it stays true for most of finalization.
bool py_is_finalizing();

// Queue a strong reference to drop later, for references a thread_local
// destructor cannot drop itself. Safe to call without the GIL.
void defer_release(nb::object&& obj);

// Drop everything queued. The caller must hold the GIL.
void drain_deferred_releases();
