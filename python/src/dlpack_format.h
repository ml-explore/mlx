// Copyright © 2026 Apple Inc.
#pragma once

#include <cstdint>

#include <nanobind/ndarray.h>

namespace nb = nanobind;

// DLPack ABI structs. We define them locally because nanobind does not expose
// the DLManagedTensor wrapper, and the patch should not introduce a new
// third-party dependency. These match the upstream DLPack header at
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
namespace dlpack_format {

struct DLManagedTensor {
  nb::dlpack::dltensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
};

struct DLPackVersion {
  uint32_t major;
  uint32_t minor;
};

struct DLManagedTensorVersioned {
  DLPackVersion version;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensorVersioned* self);
  uint64_t flags;
  nb::dlpack::dltensor dl_tensor;
};

constexpr int32_t kDLCPU = 1;
constexpr int32_t kDLCUDA = 2;
constexpr int32_t kDLMetal = 8;

} // namespace dlpack_format
