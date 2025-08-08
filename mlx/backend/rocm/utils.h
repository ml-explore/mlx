// Copyright © 2025 Apple Inc.

// This file includes utilities that are used by C++ code (i.e. .cpp files).

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {
class Device;
}

struct Dtype;

// HIP stream managed with RAII.
class HipStream {
 public:
  explicit HipStream(rocm::Device& device);
  ~HipStream();

  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;

  operator hipStream_t() const {
    return stream_;
  }

 private:
  hipStream_t stream_;
};

// Throw exception if the HIP API does not succeed.
void check_hip_error(const char* name, hipError_t err);

// The macro version that prints the command that failed.
#define CHECK_HIP_ERROR(cmd) check_hip_error(#cmd, (cmd))

// Convert Dtype to HIP C++ types.
const char* dtype_to_hip_type(const Dtype& dtype);

} // namespace mlx::core