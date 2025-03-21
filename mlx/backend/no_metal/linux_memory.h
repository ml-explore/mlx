// Copyright © 2025 Apple Inc.

#pragma once

#include <sys/sysinfo.h>

namespace {

size_t get_memory_size() {
  struct sysinfo info;

  if (sysinfo(&info) != 0) {
    return 0;
  }

  size_t total_ram = info.totalram;
  total_ram *= info.mem_unit;

  return total_ram;
}

} // namespace
