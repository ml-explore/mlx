// Copyright © 2025 Apple Inc.

#pragma once

#include <sys/sysctl.h>

namespace {

size_t get_memory_size() {
  size_t memsize = 0;
  size_t length = sizeof(memsize);
  sysctlbyname("hw.memsize", &memsize, &length, NULL, 0);
  return memsize;
}

} // namespace
