// Copyright © 2025 Apple Inc.

#pragma once

namespace {

size_t get_memory_size() {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) {
    std::cerr << "Error opening /proc/meminfo" << std::endl;
    return -1; // Indicate an error
  }

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.find("MemTotal:") == 0) {
      std::istringstream iss(line.substr(line.find(":") + 1));
      long long memTotal;
      std::string unit;
      if (iss >> memTotal >> unit) {
        if (unit == "kB") {
          return memTotal * 1024; // Convert kilobytes to bytes
        } else if (unit == "MB") {
          return memTotal * 1024 * 1024; // Convert megabytes to bytes
        } else if (unit == "GB") {
          return memTotal * 1024 * 1024 * 1024; // Convert gigabytes to bytes
        } else {
          return memTotal; // return in kilobytes if unit is unknown
        }
      }
    }
  }

  std::cerr << "MemTotal not found in /proc/meminfo" << std::endl;
  return -1; // Indicate an error
}

} // namespace
