// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/device.h"
#include "mlx/mlx_export.h"

namespace mlx::core {

struct MLX_API Stream {
  int index;
  Device device;
  explicit Stream(int index, Device device) : index(index), device(device) {}
};

/** Get the default stream for the given device. */
MLX_API Stream default_stream(Device d);

/** Make the stream the default for its device. */
MLX_API void set_default_stream(Stream s);

/** Make a new stream on the given device. */
MLX_API Stream new_stream(Device d);

/** Get the stream with the given index. */
MLX_API Stream get_stream(int index);

inline bool operator==(const Stream& lhs, const Stream& rhs) {
  return lhs.index == rhs.index;
}

inline bool operator!=(const Stream& lhs, const Stream& rhs) {
  return !(lhs == rhs);
}

/* Synchronize with the default stream. */
MLX_API void synchronize();

/* Synchronize with the provided stream. */
MLX_API void synchronize(Stream);

} // namespace mlx::core
