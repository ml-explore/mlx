// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/api.h"
#include "mlx/device.h"

namespace mlx::core {

struct MLX_API Stream {
  int index;
  Device device;
  explicit Stream(int index, Device device) : index(index), device(device) {}

  // TODO: Use default three-way comparison when it gets supported in XCode.
  bool operator==(const Stream&) const = default;
  bool operator<(const Stream& rhs) const {
    return device < rhs.device || index < rhs.index;
  }
};

/** Get the default stream of current thread for the given device. */
MLX_API Stream default_stream(Device d);

/** Make the stream the default for its device on current thread. */
MLX_API void set_default_stream(Stream s);

/** Make a new stream on the given device. */
MLX_API Stream new_stream(Device d);

/** Get all available streams. */
MLX_API std::vector<Stream> get_streams();

/* Synchronize with the default stream. */
MLX_API void synchronize();

/* Synchronize with the provided stream. */
MLX_API void synchronize(Stream);

} // namespace mlx::core
